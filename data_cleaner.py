# Create some methods for preprocessing the data
import re
import spacy

# Define some necessary methods for cleaning the text
def clean_text(text):
    
    # Remove the html tags
    html_tag_regex = '<[^<]+?>'
    new_text = re.sub(html_tag_regex,' ',text).strip()
    
    return new_text


