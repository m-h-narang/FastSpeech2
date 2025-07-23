import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
import unicodedata

import subprocess
import tempfile
import os

from torch.utils.data import DataLoader
from g2p_en import G2p
# from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence
from text import symbols


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

g2p_cache = {}

def g2p_mfa(word):
    if word in g2p_cache:
        return g2p_cache[word]

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "in.txt")
        output_path = os.path.join(tmpdir, "out.txt")
        with open(input_path, "w") as f:
            f.write(word + "\n")

        try:
            subprocess.run(
                ["mfa", "g2p", input_path, "french_mfa", output_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return ["spn"]

        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1 and parts[0].lower() == word.lower():
                        phonemes = parts[1:]
                        g2p_cache[word] = phonemes
                        return phonemes

    return ["spn"]



def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def convert_digits_to_french_words(text):
    digit_map = {
        "0": "zéro",
        "1": "un",
        "2": "deux",
        "3": "trois",
        "4": "quatre",
        "5": "cinq",
        "6": "six",
        "7": "sept",
        "8": "huit",
        "9": "neuf",
    }
    return ''.join(digit_map[c] + ' ' if c.isdigit() else c for c in text)

def join_grapheme_phonemes(phones_str):
    return "{" + " ".join(re.findall(r"\{([^{}]+)\}", phones_str)) + "}"

def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def preprocess_french(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    phones = []
    words = re.findall(r"\b[\w']+\b", text, re.UNICODE)

    for i, word in enumerate(words):
        key = word.lower()
        if key in lexicon:
            phones += lexicon[key]
        else:
            phonemes = g2p_mfa(key)
            phones += ['@' + p if not p.startswith('@') else p for p in phonemes]

        if i < len(words) - 1 and word.isalpha():
            phones.append("@sp")

    phones = [p if p.startswith('@') else '@' + p for p in phones]
    phones = [p if p in symbols else "@spn" for p in phones]
    phones_str = "{" + " ".join(p.lstrip('@') for p in phones) + "}"

    sequence = np.array(
        text_to_sequence(
            phones_str, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    if len(sequence) == 0:
        raise ValueError("Phoneme sequence is empty. Check G2P or symbol filtering.")
    print("Raw Text Sequence:", text)
    print("Phoneme Sequence:", phones_str)
    print("Final phone list:", phones)

    return sequence



def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            print("Mel Output Mean:", output[0].mean().item())
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p", "--preprocess_config", type=str, required=True, help="path to preprocess.yaml"
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.2,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=0.8,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    if args.mode == "batch":
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "fr":
            text = unicodedata.normalize("NFKC", convert_digits_to_french_words(args.text))
            texts = np.array([preprocess_french(text, preprocess_config)])
        else:
            raise ValueError("Unsupported language: {}".format(preprocess_config["preprocessing"]["text"]["language"]))

        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control
    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
