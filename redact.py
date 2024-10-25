
import spacy
nlp = spacy.load('en_core_web_lg')

def ner_redact_post_texts(p_text):
    """
    Redacts all named entities recognized by spaCy EntityRecognizer, replaces with <|PII|> pseudo-word token.
    """
    ne = list(
              [
               'PERSON',   ### people, including fictional
               'NORP',     ### nationalities or religious or political groups
               'FAC',      ### buildings, airports, highways, bridges, etc.
               'ORG',      ### companies, agencies, institutions, etc.
               #'GPE',     ### countries, cities, states
               'LOC',      ### non-GPE locations, mountain ranges, bodies of water
               'PRODUCT',  ### objects, vehicles, foods, etc. (not services)
               'EVENT',    ### named hurricanes, battles, wars, sports events, etc.
               ]
                )

    doc = nlp(p_text)
    ne_to_remove = []
    final_string = str(p_text)
    for sent in doc.ents:
        if sent.label_ in ne:
            ne_to_remove.append(str(sent.text))
    for n in range(len(ne_to_remove)):
        final_string = final_string.replace(
                                            ne_to_remove[n],
                                            '<|PII|>',
                                            )
    return final_string
