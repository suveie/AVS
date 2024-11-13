import logging
import re
from typing import Tuple

import nltk
import numpy as np
from nltk.corpus import wordnet as wn


class ObjectiveTest:
    """Class abstraction for objective test generation module."""

    def __init__(self, filepath: str):
        """Class constructor.

        Args:
            filepath (str): Absolute filepath to the subject corpus.
        """
        # Load subject corpus
        try:
            with open(filepath, mode="r") as fp:
                self.summary = fp.read()
        except FileNotFoundError:
            logging.exception("Corpus file not found.", exc_info=True)
            self.summary = ""

    def generate_test(self, num_questions: int = 3) -> Tuple[list, list]:
        """Method to generate an objective test.

        Args:
            num_questions (int, optional): Number of questions in a test.
                Defaults to 3.

        Returns:
            Tuple[list, list]: Questions and answer options respectively.
        """
        # Identify potential question sets
        question_sets = self.get_question_sets()

        # Identify potential question answers
        question_answers = []
        for question_set in question_sets:
            if question_set["Key"] > 3:
                question_answers.append(question_set)

        if len(question_answers) == 0:
            raise ValueError("No questions available in question_answers.")

        # Create objective test set
        questions, answers = [], []
        while len(questions) < num_questions:
            rand_num = np.random.randint(0, len(question_answers))
            if question_answers[rand_num]["Question"] not in questions:
                questions.append(question_answers[rand_num]["Question"])
                answers.append(question_answers[rand_num]["Answer"])
        return questions, answers

    def get_question_sets(self) -> list:
        """Method to identify sentences with potential objective questions.

        Returns:
            list: Sentences with potential objective questions.
        """
        # Tokenize corpus into sentences
        try:
            sentences = nltk.sent_tokenize(self.summary)
        except Exception:
            logging.exception("Sentence tokenization failed.", exc_info=True)
            return []

        # Identify potential question sets
        question_sets = []
        for sent in sentences:
            question_set = self.identify_potential_questions(sent)
            if question_set is not None:
                question_sets.append(question_set)
        return question_sets

    def identify_potential_questions(self, sentence: str) -> dict:
        """Method to identify potential question sets.

        Args:
            sentence (str): Tokenized sequence from corpus.

        Returns:
            dict: Question formed along with the correct answer if valid, else None.
        """
        try:
            tokens = nltk.word_tokenize(sentence)
            tags = nltk.pos_tag(tokens)

            if tags[0][1] == "RB" or len(tokens) < 4:
                return None
        except Exception:
            logging.exception("POS tagging failed.", exc_info=True)
            return None

        # Define regex grammar to chunk keywords
        noun_phrases = []
        grammar = r"""
            CHUNK: {<NN>+<IN|DT>*<NN>+}
                   {<NN>+<IN|DT>*<NNP>+}
                   {<NNP>+<NNS>*}
        """

        # Create parser tree
        chunker = nltk.RegexpParser(grammar)
        tree = chunker.parse(tags)

        # Parse tree to identify tokens
        for subtree in tree.subtrees():
            if subtree.label() == "CHUNK":
                temp = " ".join(word for word, tag in subtree.leaves())
                noun_phrases.append(temp)

        # Handle nouns
        replace_nouns = []
        for word, _ in tags:
            for phrase in noun_phrases:
                if phrase.startswith("'"):
                    break
                if word in phrase:
                    replace_nouns.extend(phrase.split()[-2:])
                    break
            if not replace_nouns:
                replace_nouns.append(word)
            break

        if not replace_nouns:
            return None

        # Find the minimum word length in replace_nouns
        val = min(len(word) for word in replace_nouns)

        trivial = {
            "Answer": " ".join(replace_nouns),
            "Key": val
        }

        if len(replace_nouns) == 1:
            trivial["Similar"] = self.answer_options(replace_nouns[0])
        else:
            trivial["Similar"] = []

        # Replace the phrase in the sentence
        replace_phrase = " ".join(replace_nouns)
        blanks_phrase = "__________" * len(replace_nouns)
        expression = re.compile(re.escape(replace_phrase), re.IGNORECASE)
        sentence = expression.sub(blanks_phrase, sentence, count=1)
        trivial["Question"] = sentence
        return trivial

    @staticmethod
    def answer_options(word: str) -> list:
        """Method to identify incorrect answer options.

        Args:
            word (str): Actual answer to the question which is to be used
                for generating other deceiving options.

        Returns:
            list: Answer options.
        """
        # In the absence of a better method, take the first synset
        try:
            synsets = wn.synsets(word, pos="n")
        except Exception:
            logging.exception("Synsets creation failed.", exc_info=True)
            return []

        # If there aren't any synsets, return an empty list
        if not synsets:
            return []

        synset = synsets[0]

        # Get the hypernym for this synset
        hypernyms = synset.hypernyms()
        if not hypernyms:
            return []

        hypernym = hypernyms[0]

        # Get some hyponyms from this hypernym
        hyponyms = hypernym.hyponyms()

        # Take the name of the first lemma for the first 8 hyponyms
        similar_words = [
            hyponym.lemmas()[0].name().replace("_", " ")
            for hyponym in hyponyms
            if hyponym.lemmas()[0].name().replace("_", " ") != word
        ][:8]

        return similar_words
