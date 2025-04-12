import os
from pathlib import Path
import numpy as np
import torch

from src.app import CONFIG_PATH


class CTCLabelConverter:
    """Convert between text-label and text-index for Latin languages"""

    def __init__(self, character, lang_list: list):
        """
        Initialize the CTC Label Converter for Latin languages.

        Args:
            character (str): Set of the possible characters.
            dictionary_path (dict[str, Path]): Dictionary containing language to dictionary file path mapping.
        """
        dict_character = list(character)
        dictionary_path = {}
        for lang in lang_list:
            dictionary_path[lang] = os.path.join(
                CONFIG_PATH, "lang_dicts", lang + ".txt"
            )

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1

        # Add dummy '[blank]' token for CTCLoss (index 0)
        self.character = ["[blank]"] + dict_character

        dict_list = []
        for lang, dict_path in dictionary_path.items():
            try:
                with open(dict_path, "r", encoding="utf-8-sig") as input_file:
                    word_count = input_file.read().splitlines()
                dict_list += word_count
            except:
                pass

        self.dict_list = dict_list
        self.ignore_idx = [0]  # Only ignore the blank token

    def encode(self, text):
        """
        Convert text-label into text-index.

        Args:
            text: Text labels of each image. [batch_size]

        Returns:
            tuple: (text, length)
                - text: Concatenated text index for CTCLoss.
                      [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
                - length: Length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = "".join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode_greedy(self, text_index, length):
        """
        Convert text-index into text-label using greedy decoding.

        Args:
            text_index: Text index from output of the model. [sum(length)]
            length: Length of each text. [batch_size]

        Returns:
            list: List of decoded texts.
        """
        texts = []
        index = 0
        for l in length:
            t = text_index[index : index + l]

            # Returns a boolean array where true is when the value is not repeated
            a = np.insert(~((t[1:] == t[:-1])), 0, True)

            # Returns a boolean array where true is when the value is not the blank token
            b = ~np.isin(t, np.array(self.ignore_idx))

            # Combine the two boolean arrays
            c = a & b

            # Gets the corresponding character according to the saved indexes
            text = "".join(np.array(self.character)[t[c.nonzero()]])
            texts.append(text)
            index += l

        return texts
