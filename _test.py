from llm_handler import analyse_sentiments

def test():
    test_strings = [
        "Migrants have significantly contributed to the UK economy.",
        "This article highlights the positive cultural impact of migration.",
        "The new policy supports integrating migrants into the workforce.",
        "Local communities have welcomed migrants with open arms.",
        "Migration enriches society and brings diversity.",
        "Migrants are putting pressure on public services.",
        "The report blames rising crime rates on recent migration.",
        "This article portrays migrants as a threat to national identity.",
        "Illegal immigration is described as out of control.",
        "Migrants are accused of taking jobs from locals.",
        "The government released new migration statistics today.",
        "The article outlines changes in migration laws without opinion.",
        "A timeline of major UK migration events is provided.",
        "The piece discusses both benefits and challenges of migration.",
        "Migration numbers have steadily increased since 2010.",
        "While migrants help the economy, the article also raises concerns about integration.",
        "The headline praises diversity, but the tone feels cautious.",
        "Quotes from experts are supportive, but reader comments are critical.",
        "The article presents contrasting views on immigration policy.",
        "Migration is described as a complex and divisive issue."
    ]

    choices = ["positive", "negative", "neutral"]

    prompt = """
    The sentence is taken from a newspaper article on migration, 
    in what light does it portray migrants, immigrants, asylum seekers, or ethnic minorities? 
    """

    output_sentiments = analyse_sentiments(test_strings, prompt, choices, model="deepseek-r1:1.5b", max_workers=4, debug=True)

    result = zip(test_strings, output_sentiments)

    for r in result:
        print(tuple(r))


if __name__ == "__main__":
    test()
