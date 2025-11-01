import re
import nltk

from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer


class ScraperHelper():

    def __init__(self):
        """
        this thing should be only initialized once before looping every document
        in order to clean it
        """
        # Download stopwords if not already downloaded
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        # Define stopwords but keep critical ones like "not"
        base_stopwords = set(stopwords.words('english'))
        preserve = {'no', 'not', 'nor', 'never'}
        self.custom_stopwords = base_stopwords - preserve

        self.lemmatizer = WordNetLemmatizer()
        self.emoji_pattern = (
            re.compile("["

                       # Emoticons (e.g., 😀😁😂🤣😃😄😅😆)
                       u"\U0001F600-\U0001F64F"

                       # Symbols & pictographs (e.g., 🔥🎉💡📦📱)
                       u"\U0001F300-\U0001F5FF"

                       # Transport & map symbols (e.g., 🚗✈️🚀🚉)
                       u"\U0001F680-\U0001F6FF"

                       # Flags (e.g., 🇺🇸🇬🇧🇨🇦 — these are pairs of regional indicators)
                       u"\U0001F1E0-\U0001F1FF"

                       # Dingbats (e.g., ✂️✈️✉️⚽)
                       u"\u2700-\u27BF"

                       # Supplemental Symbols & Pictographs (e.g., 🤖🥰🧠🦾)
                       u"\U0001F900-\U0001F9FF"

                       # Symbols & Pictographs Extended-A (e.g., 🪄🪅🪨)
                       u"\U0001FA70-\U0001FAFF"

                       # Miscellaneous symbols (e.g., ☀️☁️☂️⚡)
                       u"\u2600-\u26FF"

                       "]+", flags=re.UNICODE))

        # This pattern will match common text-based emoticons that aren't covered by the emoji Unicode ranges
        # These emoticons are made up of regular ASCII characters like colons, parentheses, etc.

        self.emoticon_pattern = re.compile(r'(:\)|:\(|:D|:P|;\)|:-\)|:-D|:-P|:\'\(|:\||:\*)')

    def lowercase_text(self, text):
        # Convert text to lowercase.
        return str(text).lower()



    def replace_urls(self, text: str):
        """
        Replace URLs in the text with the token 'URL'.
        Prints before and after if a replacement occurs.
        """
        # Define a regex pattern to identify URLs in the text
        url_pattern = r"(?:https?|ftp)://[^\s/$.?#].[^\s]*"
        text_str = str(text)
        replaced_text = re.sub(url_pattern, 'URL', text_str)
        return replaced_text

    # re.compile will compile the regex pattern into a regex object, necessary for
    # efficient pattern matching. This creates a reusable pattern object that can be
    # used multiple times without recompiling the pattern each time, improving performance.
    # u stands for Unicode

    def remove_and_print(self, text):
        if self.emoji_pattern.search(text) or self.emoticon_pattern.search(text):
            print(f"Before: {text}")
            text = self.emoji_pattern.sub('', text)
            text = self.emoticon_pattern.sub('', text)
            print(f"After: {text}")
            print()
        return text

    def replace_usernames(self, text):
        """
        Replace email addresses and true @usernames with 'USER'.
        Avoid matching embedded @ in profanity or stylized words.
        Print before and after if replacement occurs.
        """
        original = str(text)
        updated = original

        # Replace full email addresses
        updated = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', 'USER', updated)

        # Replace @usernames only when preceded by space, punctuation, or start of string
        updated = re.sub(r'(?:(?<=^)|(?<=[\s.,;!?]))@\w+\b', 'USER', updated)

        return updated


    def clean_text(self, text, keep_punct=False):
        """
        Clean and normalize text for NLP classification tasks.

        Parameters:
        - text (str): The input text to be cleaned.
        - keep_punct (bool):
            If True, retains key punctuation (. ! ?) which may carry emotional or contextual weight.
            If False, removes all non-alphabetic characters for simpler lexical analysis.

        Returns:
        - str: The cleaned text string, lowercased and stripped of unwanted characters.

        This function is designed for flexibility across different NLP tasks like sentiment analysis,
        topic classification, or spam detection. It handles:
        - Lowercasing text for normalization
        - Removing or preserving select punctuation
        - Removing digits, symbols, and special characters
        - Reducing multiple spaces to a single space
        - Optionally printing changes for debugging or logging

        When to use `keep_punct=True`:
        - Sentiment analysis: punctuation (e.g., "!", "?") can reflect strong emotion
        - Social media or informal text: expressive punctuation often carries signal
        - Sarcasm, emphasis, or tone-sensitive tasks

        When to use `keep_punct=False`:
        - Topic classification or document clustering: punctuation rarely adds value
        - Preprocessing for bag-of-words, TF-IDF, or topic modeling
        - When punctuation is inconsistent or noisy (e.g., OCR scans, scraped data)
        """

        # Convert input to string (safe handling)
        original = str(text)

        if keep_punct:
            # Keep only lowercase letters, spaces, and select punctuation (. ! ?)
            # Useful for capturing tone/sentiment
            cleaned = re.sub(r"[^a-z\s.!?]", "", original)
        else:
            # Keep only lowercase letters and spaces; remove all punctuation and symbols
            cleaned = re.sub(r"[^a-z\s]", "", original)

        # Normalize whitespace (collapse multiple spaces to one, strip leading/trailing)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Optional: print before and after if a change occurred
        return cleaned

    def remove_consecutive_letters(self, text, max_repeat=2):
        """
        Normalize elongated words by limiting repeated characters.

        In informal or emotional text (e.g., reviews, tweets), users often repeat letters
        to add emphasis: "sooooo good", "loooove it", "greeaaat".

        This function reduces any character repeated more than `max_repeat` times
        to exactly `max_repeat` occurrences (default: 2), preserving emphasis without bloating vocabulary.

        Parameters:
        - text (str): The input text
        - max_repeat (int): The maximum allowed repetitions for any character

        Returns:
        - str: Text with repeated characters normalized
        """
        text_str = str(text)
        pattern = r'(\w)\1{' + str(max_repeat) + r',}'
        cleaned = re.sub(pattern, r'\1' * max_repeat, text_str)

        return cleaned

    def remove_short_words(self, text, min_length=3, preserve_words=None):
        """
        Remove short words from text based on a minimum length threshold.

        Parameters:
        - text (str): The input text
        - min_length (int): Minimum word length to keep (default = 3)
        - preserve_words (set or list): Optional set of short but important words to keep (e.g., {'no', 'not'})

        Returns:
        - str: Text with short words removed, except for preserved ones

        Notes:
        - Use with care in sentiment analysis. Important short words like 'no', 'not', 'bad' may affect meaning.
        - Best used after stopword removal or on very noisy text.
        """
        preserve = set(preserve_words or [])
        words = str(text).split()
        filtered = [word for word in words if len(word) >= min_length or word.lower() in preserve]
        result = ' '.join(filtered)

        return result

    def remove_stopwords(self, text):
        """
        Remove stopwords from text, preserving key negation words.

        This function uses a customized stopword list that retains important
        short words like 'not', 'no', 'nor', and 'never' which carry significant
        meaning in tasks like sentiment analysis.

        Parameters:
        - text (str): Lowercased input text

        Returns:
        - str: Text with stopwords removed, but critical negation words preserved
        """
        words = str(text).split()
        filtered = [word for word in words if word not in self.custom_stopwords]
        result = ' '.join(filtered)

        return result

    # Download required NLTK resources
    nltk.download('wordnet')  # Download WordNet, a lexical database of English words
    nltk.download(
        'omw-1.4')  # WordNet Lemmas sometimes need this, which is a mapping of WordNet lemmas to their Part of Speech (POS) tags.
    nltk.download('averaged_perceptron_tagger_eng')  # Download English POS tagger


    # POS mapping function
    # POS tags can be: ADJ (adjective), ADV (adverb), NOUN (noun), VERB (verb), etc
    def get_wordnet_pos(self, tag):
        # Determine the WordNet POS tag based on the first letter of the input tag
        if tag.startswith('J'):
            return wordnet.ADJ  # Adjective
        elif tag.startswith('V'):
            return wordnet.VERB  # Verb
        elif tag.startswith('N'):
            return wordnet.NOUN  # Noun
        elif tag.startswith('R'):
            return wordnet.ADV  # Adverb
        else:
            return wordnet.NOUN  # Default to Noun if no match

    def lemmatize_text(self, text):
        """
        Lemmatize text using WordNet lemmatizer with POS tagging.

        This version prints each change along with the POS tag of the changed word.
        """
        # Convert the input text to a string to ensure compatibility
        original_text = str(text)
        # Split the text into individual words
        words = original_text.split()
        # Obtain Part of Speech (POS) tags for each word
        pos_tags = pos_tag(words)

        # Initialize lists to store lemmatized words and any changes
        lemmatized_words = []
        changes = []

        # Iterate over each word and its POS tag
        for word, tag in pos_tags:
            # Map the POS tag to a WordNet POS tag
            wn_tag = self.get_wordnet_pos(tag)
            # Lemmatize the word using the mapped POS tag
            lemma = self.lemmatizer.lemmatize(word, wn_tag)

            # Check if the lemmatized word is different from the original
            if lemma != word:
                # Record the change if a difference is found
                changes.append((word, lemma, tag))
            # Add the lemmatized word to the list
            lemmatized_words.append(lemma)

        # Join the lemmatized words back into a single string
        result = ' '.join(lemmatized_words)
        return result

    def remove_punctuation(self, text):
        """
        Removes all punctuation from the text, keeping only letters, digits, and spaces.
        Also collapses extra spaces and strips leading/trailing whitespace.
        """
        text = re.sub(r'[^\w\s]', '', str(text))  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
        return text.strip()
