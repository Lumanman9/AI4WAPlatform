import spacy
import json

class EventExtractor:
    def __init__(self, model="en_core_web_sm"):
        """Initialize the EventExtractor with a specified SpaCy model."""
        self.nlp = spacy.load(model)

    def process_text(self, text):
        """Process the input text and extract events and tokens."""
        # Process text with SpaCy
        doc = self.nlp(text)

        # Initialize data structure for JSON output
        result = []
        sentence_id = 0

        # Process each sentence
        for sent in doc.sents:
            sentence_data = {
                "sentenceID": sentence_id,
                "sentence": sent.text,
                "tokens": [],
                "events": []
            }

            token_id = 0
            for token in sent:
                sentence_data["tokens"].append({
                    "tokenID": token_id,
                    "token": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "dep": token.dep_
                })
                token_id += 1

            # Extract events (root verb and related entities)
            root = [token for token in sent if token.head == token]  # Root verb
            if root:
                root_token = root[0]
                event_data = {
                    "event": root_token.text,
                    "eventTokenID": root_token.i - sent.start  # Token ID relative to sentence
                }
                sentence_data["events"].append(event_data)

            result.append(sentence_data)
            sentence_id += 1

        return result

    def extract_events_from_file(self, input_file, output_file):
        """Extract events from an input file and write them to an output file in JSON format."""
        # Read input text file
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()

        # Process the text
        result = self.process_text(text)

        # Write output to JSON file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(result, outfile, indent=4, ensure_ascii=False)

        print(f"Event extraction completed. Results saved to {output_file}")


# Example usage
if __name__ == "__main__":
    input_file = "/Users/manqin/PycharmProjects/ECKGs/datasets/input_text.txt"  # Replace with your input text file
    output_file = "output_events.json"  # Replace with your desired output file

    extractor = EventExtractor()
    extractor.extract_events_from_file(input_file, output_file)
