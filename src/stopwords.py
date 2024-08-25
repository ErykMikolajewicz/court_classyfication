# Stopwords from: https://www.ranks.nl/stopwords/polish
_basic = {
    'ach',
    'aj',
    'albo',
    'bardzo',
    'bez',
    'bo',
    'być',
    'ci',
    'cię',
    'ciebie',
    'co',
    'czy',
    'daleko',
    'dla',
    'dlaczego',
    'dlatego',
    'do',
    'dobrze',
    'dokąd',
    'dość',
    'dużo',
    'dwa',
    'dwaj',
    'dwie',
    'dwoje',
    'dziś',
    'dzisiaj',
    'gdyby',
    'gdzie',
    'go',
    'ich',
    'ile',
    'im',
    'inny',
    'ja',
    'ją',
    'jak',
    'jakby',
    'jaki',
    'je',
    'jeden',
    'jedna',
    'jedno',
    'jego',
    'jej',
    'jemu',
    'jeśli',
    'jest',
    'jestem',
    'jeżeli',
    'już',
    'każdy',
    'kiedy',
    'kierunku',
    'kto',
    'ku',
    'lub',
    'ma',
    'mają',
    'mam',
    'mi',
    'mną',
    'mnie',
    'moi',
    'mój',
    'moja',
    'moje',
    'może',
    'mu',
    'my',
    'na',
    'nam',
    'nami',
    'nas',
    'nasi',
    'nasz',
    'nasza',
    'nasze',
    'natychmiast',
    'nią',
    'nic',
    'nich',
    'nie',
    'niego',
    'niej',
    'niemu',
    'nigdy',
    'nim',
    'nimi',
    'niż',
    'obok',
    'od',
    'około',
    'on',
    'ona',
    'one',
    'oni',
    'ono',
    'owszem',
    'po',
    'pod',
    'ponieważ',
    'przed',
    'przedtem',
    'są',
    'sam',
    'sama',
    'się',
    'skąd',
    'tak',
    'taki',
    'tam',
    'ten',
    'to',
    'tobą',
    'tobie',
    'tu',
    'tutaj',
    'twoi',
    'twój',
    'twoja',
    'twoje',
    'ty',
    'wam',
    'wami',
    'was',
    'wasi',
    'wasz',
    'wasza',
    'wasze',
    'we',
    'więc',
    'wszystko',
    'wtedy',
    'wy',
    'żaden',
    'zawsze',
    'że',
}

_my_basic = {
    'na',
    'w',
    'z',
    'o',
    'do',
    'nie',
    'art',
    'się',
    'co',
    'od',
    'przez',
    'że',
    'a',
    'za',
    'oraz',
    'po',
    'ze',
    'dla',
    'przy',
    'może',
    'iż',
    'we'
}

# Words which make problems for stemmer, not only make invalid output, but also make words, that are other words.
# Usually short not very meaningful words
_stemmer_problematic = {
    'a',  # ativus
    'c',  # córka
    'czego',  # co
    'czym',  # co
    'da',  # dać
    'in',  # inny ? should not be word such as in, need check
    'mr',  # mrok
    'o',  # ojciec
    'od',  # oda
    'te',  # ty
    'tych',  # en  ? very unexpected stemming result
    'w',  # wyspa
    'ze',  # z
    'życie',  # żyto
    'v',  # vocativus
    'm',  # morze
    'hr',  # hrabia
    'pole',  # poła
    'szt',  # sztuka
}

before_stemming = _basic.union(_stemmer_problematic)
before_stemming = before_stemming.union(_my_basic)


# Thing I decided to not include for my knowledge and intuition
domain_knowledge = {
    # To avoid not consistency by not using digits to word remove months names,
    # to not make distinguish between text and numeral date format
    'styczeń',
    'luty',
    'marzec',
    'marc',  # stemmer error
    'kwiecień'
    'maj',
    'czerwiec',
    'lipiec',
    'sierpień',
    'wrzesień',
    'październik',
    'listopad',
    'grudzień',
    'r',
    #  Stemmer work very poorly with versions of word który and made from it very various other words,
    #  not all captured by me
    'który',
    'któyy',
    # It is a Python None, strangely stemmer sometimes produce it as output.
    'None'

}

# Words with frequency more than 90% in all texts perhaps in should decrease frequency threshold.
_my_basic_after_stemming = {
    'sąd',
    'dzień',
    'to',
    'zważyć',
    'ustalić',
    'stan',
    'faktyczny',
    'następować',
    'strona',
    'były',
    'zgodni',
    'podstawa',
    'mieć',
    'który',
    'powyższy',
    'wnieść',
    'dowód'
}

#  Words to remove after stemming, to get gain from reduce number of word versions
after_stemming = domain_knowledge.union(_my_basic_after_stemming)
