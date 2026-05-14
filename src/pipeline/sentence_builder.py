# sentence_builder.py
# -------------------------------------------------------
# What this does:
#   Takes the smoother's accepted outputs one at a time
#   and builds a sentence from them.
#
#   It handles:
#     - regular letters → spell out a word
#     - "space"         → finish current word, add a space
#     - "del"           → delete last character
#     - "clear"         → wipe the whole sentence
#     - whole words     → add directly (from phrase model)
# -------------------------------------------------------

class SentenceBuilder:
    def __init__(self):
        self.sentence = ""   # the raw sentence so far

    def add(self, token: str):
        """
        Add one accepted token to the sentence.
        token can be: a letter ("A"), a word ("HELLO"), or a command ("space", "del", "clear")

        Returns the current sentence after the update.
        """

        # --- commands ---
        if token.lower() == "space":
            self.sentence += " "

        elif token.lower() == "del":
            self.sentence = self.sentence[:-1]   # remove last character

        elif token.lower() == "clear":
            self.sentence = ""                   # wipe everything

        # --- regular letter or whole word ---
        else:
            self.sentence += token               # just append it

        return self.sentence

    def get(self):
        """Returns the current raw sentence."""
        return self.sentence

    def reset(self):
        """Wipe the sentence."""
        self.sentence = ""
