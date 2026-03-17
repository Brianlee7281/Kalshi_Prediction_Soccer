"""Team name alias mapping across data sources.

Maps canonical team names to all known aliases (lowercased) from:
- Goalserve (data/commentaries/)
- football-data.co.uk (data/odds_historical/)
- Odds-API.io (official long names with city/state suffixes)
- Kalshi (abbreviated names from market titles)

Usage:
    from src.calibration.team_aliases import normalize_team_name
    canonical = normalize_team_name("Man City")  # -> "manchester city"
"""

import unicodedata


# Canonical name -> list of lowercased aliases (including canonical itself lowered)
TEAM_ALIASES: dict[str, list[str]] = {
    # =========================================================================
    # EPL (1204)
    # =========================================================================
    "AFC Bournemouth": [
        "afc bournemouth",
        "bournemouth",
    ],
    "Arsenal": [
        "arsenal",
        "arsenal fc",
    ],
    "Aston Villa": [
        "aston villa",
    ],
    "Brentford": [
        "brentford",
        "brentford fc",
    ],
    "Brighton": [
        "brighton",
        "brighton & hove albion",
        "brighton and hove albion",
    ],
    "Burnley": [
        "burnley",
        "burnley fc",
    ],
    "Chelsea": [
        "chelsea",
        "chelsea fc",
    ],
    "Crystal Palace": [
        "crystal palace",
    ],
    "Everton": [
        "everton",
        "everton fc",
    ],
    "Fulham": [
        "fulham",
        "fulham fc",
    ],
    "Ipswich Town": [
        "ipswich town",
        "ipswich t.",
        "ipswich",
    ],
    "Leeds United": [
        "leeds united",
        "leeds",
    ],
    "Leicester City": [
        "leicester city",
        "leicester",
    ],
    "Liverpool": [
        "liverpool",
        "liverpool fc",
    ],
    "Luton Town": [
        "luton town",
        "luton",
    ],
    "Manchester City": [
        "manchester city",
        "man city",
    ],
    "Manchester United": [
        "manchester united",
        "man united",
        "manchester utd",
    ],
    "Newcastle United": [
        "newcastle united",
        "newcastle",
    ],
    "Norwich City": [
        "norwich city",
        "norwich",
    ],
    "Nottingham Forest": [
        "nottingham forest",
        "nott'm forest",
        "nottingham",
    ],
    "Sheffield United": [
        "sheffield united",
    ],
    "Southampton": [
        "southampton",
    ],
    "Tottenham Hotspur": [
        "tottenham hotspur",
        "tottenham",
    ],
    "Watford": [
        "watford",
    ],
    "West Bromwich Albion": [
        "west bromwich albion",
        "west brom",
    ],
    "West Ham United": [
        "west ham united",
        "west ham",
    ],
    "Sunderland": [
        "sunderland",
        "sunderland afc",
    ],
    "Wolverhampton": [
        "wolverhampton",
        "wolverhampton wanderers",
        "wolves",
    ],

    # =========================================================================
    # La Liga (1399)
    # =========================================================================
    "Almeria": [
        "almeria",
        "ud almeria",
    ],
    "Athletic Club": [
        "athletic club",
        "athletic bilbao",
        "ath bilbao",
        "bilbao",
    ],
    "Atletico Madrid": [
        "atletico madrid",
        "atletico de madrid",
        "ath madrid",
        "atletico",
        "atl. madrid",
    ],
    "Barcelona": [
        "barcelona",
        "fc barcelona",
    ],
    "Cadiz": [
        "cadiz",
        "cadiz cf",
    ],
    "Celta Vigo": [
        "celta vigo",
        "celta de vigo",
        "celta",
        "rc celta de vigo",
    ],
    "Deportivo Alaves": [
        "deportivo alaves",
        "alaves",
    ],
    "Eibar": [
        "eibar",
        "sd eibar",
    ],
    "Elche": [
        "elche",
        "elche cf",
    ],
    "Espanyol": [
        "espanyol",
        "espanol",
        "rcd espanyol",
        "espanyol barcelona",
    ],
    "Getafe": [
        "getafe",
        "getafe cf",
    ],
    "Girona": [
        "girona",
        "girona fc",
    ],
    "Granada": [
        "granada",
        "granada cf",
    ],
    "Huesca": [
        "huesca",
        "sd huesca",
    ],
    "Las Palmas": [
        "las palmas",
        "ud las palmas",
    ],
    "Leganes": [
        "leganes",
        "cd leganes",
    ],
    "Levante": [
        "levante",
        "levante ud",
    ],
    "Mallorca": [
        "mallorca",
        "rcd mallorca",
    ],
    "Osasuna": [
        "osasuna",
        "ca osasuna",
    ],
    "Rayo Vallecano": [
        "rayo vallecano",
        "vallecano",
    ],
    "Real Betis": [
        "real betis",
        "betis",
        "real betis seville",
    ],
    "Real Madrid": [
        "real madrid",
    ],
    "Real Sociedad": [
        "real sociedad",
        "sociedad",
        "real sociedad san sebastian",
    ],
    "Real Valladolid": [
        "real valladolid",
        "valladolid",
    ],
    "Sevilla": [
        "sevilla",
        "sevilla fc",
    ],
    "Valencia": [
        "valencia",
        "valencia cf",
    ],
    "Real Oviedo": [
        "real oviedo",
        "oviedo",
    ],
    "Villarreal": [
        "villarreal",
        "villarreal cf",
    ],

    # =========================================================================
    # Serie A (1269)
    # =========================================================================
    "Atalanta": [
        "atalanta",
        "atalanta bc",
    ],
    "Benevento": [
        "benevento",
    ],
    "Bologna": [
        "bologna",
        "bologna fc",
    ],
    "Brescia": [
        "brescia",
    ],
    "Cagliari": [
        "cagliari",
        "cagliari calcio",
    ],
    "Como": [
        "como",
        "como 1907",
    ],
    "Cremonese": [
        "cremonese",
        "us cremonese",
    ],
    "Crotone": [
        "crotone",
    ],
    "Empoli": [
        "empoli",
    ],
    "Fiorentina": [
        "fiorentina",
        "acf fiorentina",
    ],
    "Frosinone": [
        "frosinone",
    ],
    "Genoa": [
        "genoa",
        "genoa cfc",
    ],
    "Hellas Verona": [
        "hellas verona",
        "verona",
    ],
    "Inter Milan": [
        "inter milan",
        "inter",
        "internazionale",
        "inter milano",
    ],
    "Juventus": [
        "juventus",
        "juventus turin",
    ],
    "Lazio": [
        "lazio",
        "lazio rome",
    ],
    "Lecce": [
        "lecce",
        "us lecce",
    ],
    "Milan": [
        "milan",
        "ac milan",
    ],
    "Monza": [
        "monza",
    ],
    "Napoli": [
        "napoli",
        "ssc napoli",
    ],
    "Parma": [
        "parma",
        "parma calcio",
    ],
    "Roma": [
        "roma",
        "as roma",
    ],
    "Salernitana": [
        "salernitana",
    ],
    "Sampdoria": [
        "sampdoria",
    ],
    "Sassuolo": [
        "sassuolo",
        "sassuolo calcio",
    ],
    "SPAL": [
        "spal",
        "real spal",
    ],
    "Spezia": [
        "spezia",
    ],
    "Pisa": [
        "pisa",
        "pisa sc",
    ],
    "Torino": [
        "torino",
        "torino fc",
    ],
    "Udinese": [
        "udinese",
        "udinese calcio",
    ],
    "Venezia": [
        "venezia",
    ],

    # =========================================================================
    # Bundesliga (1229)
    # =========================================================================
    "FC Cologne": [
        "fc cologne",
        "1. fc cologne",
        "fc koln",
        "1. fc koln",
        "koln",
    ],
    "Arminia Bielefeld": [
        "arminia bielefeld",
        "bielefeld",
    ],
    "Augsburg": [
        "augsburg",
        "fc augsburg",
    ],
    "Bayer Leverkusen": [
        "bayer leverkusen",
        "leverkusen",
    ],
    "Bayern Munich": [
        "bayern munich",
        "bayern munchen",
        "fc bayern",
    ],
    "Bochum": [
        "bochum",
        "vfl bochum",
    ],
    "Borussia Monchengladbach": [
        "borussia monchengladbach",
        "borussia m'gladbach",
        "m'gladbach",
        "monchengladbach",
        "gladbach",
        "m gladbach",
        "b. monchengladbach",
    ],
    "Darmstadt 98": [
        "darmstadt 98",
        "darmstadt",
        "sv darmstadt 98",
    ],
    "Borussia Dortmund": [
        "borussia dortmund",
        "dortmund",
    ],
    "Eintracht Frankfurt": [
        "eintracht frankfurt",
        "ein frankfurt",
        "frankfurt",
    ],
    "Fortuna Dusseldorf": [
        "fortuna dusseldorf",
        "fortuna dusseldorf",
    ],
    "Freiburg": [
        "freiburg",
        "sc freiburg",
    ],
    "Greuther Furth": [
        "greuther furth",
        "greuther fuerth",
    ],
    "Hamburger SV": [
        "hamburger sv",
        "hamburg",
    ],
    "Heidenheim": [
        "heidenheim",
        "1. fc heidenheim",
    ],
    "Hertha BSC": [
        "hertha bsc",
        "hertha",
        "hertha berlin",
    ],
    "Hoffenheim": [
        "hoffenheim",
        "tsg hoffenheim",
    ],
    "Holstein Kiel": [
        "holstein kiel",
        "kiel",
    ],
    "Mainz 05": [
        "mainz 05",
        "mainz",
        "fsv mainz",
        "1. fsv mainz 05",
    ],
    "Paderborn": [
        "paderborn",
        "sc paderborn",
    ],
    "RB Leipzig": [
        "rb leipzig",
        "leipzig",
    ],
    "Schalke 04": [
        "schalke 04",
        "schalke",
    ],
    "St Pauli": [
        "st pauli",
        "fc st. pauli",
        "st. pauli",
    ],
    "Stuttgart": [
        "stuttgart",
        "vfb stuttgart",
    ],
    "Union Berlin": [
        "union berlin",
        "1. fc union berlin",
    ],
    "Werder Bremen": [
        "werder bremen",
        "bremen",
    ],
    "Wolfsburg": [
        "wolfsburg",
        "vfl wolfsburg",
    ],

    # =========================================================================
    # Ligue 1 (1221)
    # =========================================================================
    "Ajaccio": [
        "ajaccio",
        "ac ajaccio",
    ],
    "Amiens": [
        "amiens",
        "amiens sc",
    ],
    "Angers": [
        "angers",
        "angers sco",
    ],
    "Auxerre": [
        "auxerre",
        "aj auxerre",
    ],
    "Bordeaux": [
        "bordeaux",
        "girondins de bordeaux",
    ],
    "Brest": [
        "brest",
        "stade brestois",
        "stade brest",
        "stade brest 29",
    ],
    "Clermont Foot": [
        "clermont foot",
        "clermont",
    ],
    "Dijon": [
        "dijon",
        "dijon fco",
    ],
    "Le Havre": [
        "le havre",
        "le havre ac",
    ],
    "Lens": [
        "lens",
        "rc lens",
        "racing club de lens",
    ],
    "Lille": [
        "lille",
        "losc lille",
        "lille osc",
    ],
    "Lorient": [
        "lorient",
        "fc lorient",
    ],
    "Lyon": [
        "lyon",
        "olympique lyonnais",
        "olympique lyon",
    ],
    "Marseille": [
        "marseille",
        "olympique de marseille",
        "olympique marseille",
    ],
    "Metz": [
        "metz",
        "fc metz",
    ],
    "Monaco": [
        "monaco",
        "as monaco",
    ],
    "Montpellier": [
        "montpellier",
        "montpellier hsc",
    ],
    "Nantes": [
        "nantes",
        "fc nantes",
    ],
    "Nice": [
        "nice",
        "ogc nice",
    ],
    "Nimes": [
        "nimes",
        "nimes olympique",
    ],
    "Paris FC": [
        "paris fc",
        "paris",
    ],
    "Paris Saint-Germain": [
        "paris saint-germain",
        "paris s.g.",
        "paris sg",
        "psg",
    ],
    "Reims": [
        "reims",
        "stade de reims",
    ],
    "Rennes": [
        "rennes",
        "stade rennais",
        "stade rennais fc",
    ],
    "Saint-Etienne": [
        "saint-etienne",
        "st etienne",
        "as saint-etienne",
    ],
    "Strasbourg": [
        "strasbourg",
        "rc strasbourg",
        "strasbourg alsace",
    ],
    "Toulouse": [
        "toulouse",
        "toulouse fc",
    ],
    "Troyes": [
        "troyes",
        "estac troyes",
    ],

    # =========================================================================
    # MLS (1440)
    # =========================================================================
    "Atlanta United": [
        "atlanta united",
        "atlanta united fc",
        "atlanta utd",
        "atlanta",
    ],
    "Austin FC": [
        "austin fc",
        "austin",
    ],
    "Charlotte FC": [
        "charlotte fc",
        "charlotte",
    ],
    "Chicago Fire": [
        "chicago fire",
        "chicago fire fc",
        "chicago",
    ],
    "FC Cincinnati": [
        "fc cincinnati",
        "cincinnati",
    ],
    "Colorado Rapids": [
        "colorado rapids",
        "colorado",
    ],
    "Columbus Crew": [
        "columbus crew",
        "columbus",
    ],
    "DC United": [
        "dc united",
        "d.c. united",
    ],
    "FC Dallas": [
        "fc dallas",
        "dallas",
    ],
    "Houston Dynamo": [
        "houston dynamo",
        "houston",
    ],
    "Inter Miami": [
        "inter miami",
        "inter miami cf",
        "miami",
    ],
    "LA Galaxy": [
        "la galaxy",
        "los angeles galaxy",
        "los angeles g",
    ],
    "Los Angeles FC": [
        "los angeles fc",
        "lafc",
        "los angeles f",
    ],
    "Minnesota United": [
        "minnesota united",
        "minnesota united fc",
        "minnesota",
    ],
    "CF Montreal": [
        "cf montreal",
        "montreal impact",
        "montreal",
    ],
    "Nashville SC": [
        "nashville sc",
        "nashville",
    ],
    "New England Revolution": [
        "new england revolution",
        "new england re'lution",
        "new england",
    ],
    "New York City FC": [
        "new york city fc",
        "new york city",
        "nycfc",
    ],
    "New York Red Bulls": [
        "new york red bulls",
        "new york rb",
        "ny red bulls",
        "new york",
    ],
    "Orlando City": [
        "orlando city",
        "orlando city sc",
        "orlando",
    ],
    "Philadelphia Union": [
        "philadelphia union",
        "philadelphia",
    ],
    "Portland Timbers": [
        "portland timbers",
        "portland",
    ],
    "Real Salt Lake": [
        "real salt lake",
        "rsl",
        "salt lake",
    ],
    "San Jose Earthquakes": [
        "san jose earthquakes",
        "sj earthquakes",
        "san jose",
    ],
    "San Diego FC": [
        "san diego fc",
        "san diego",
    ],
    "Seattle Sounders": [
        "seattle sounders",
        "seattle sounders fc",
        "seattle",
    ],
    "Sporting Kansas City": [
        "sporting kansas city",
        "sporting kc",
        "kansas city",
    ],
    "St. Louis City": [
        "st. louis city",
        "st louis city",
        "st. louis city sc",
        "saint louis city sc",
        "saint louis",
    ],
    "Toronto FC": [
        "toronto fc",
        "toronto",
    ],
    "Vancouver Whitecaps": [
        "vancouver whitecaps",
        "vancouver whitecaps fc",
        "vancouver",
    ],
    "MLS All-Stars": [
        "mls all-stars",
    ],
    "Liga MX All-Stars": [
        "liga mx all-stars",
    ],
    "Chivas USA": [
        "chivas usa",
    ],

    # =========================================================================
    # Brasileirao (1141)
    # =========================================================================
    "America Mineiro": [
        "america mineiro",
        "america mg",
    ],
    "Athletico Paranaense": [
        "athletico paranaense",
        "athletico-pr",
        "atletico paranaense",
        "paranaense",
        "ca paranaense pr",
        "ca paranaense",
    ],
    "Atletico Goianiense": [
        "atletico goianiense",
        "atletico go",
    ],
    "Atletico Mineiro": [
        "atletico mineiro",
        "atletico-mg",
        "atletico mineiro mg",
    ],
    "Avai": [
        "avai",
        "avai fc",
    ],
    "Bahia": [
        "bahia",
        "ec bahia",
        "ec bahia ba",
    ],
    "Botafogo": [
        "botafogo",
        "botafogo rj",
        "botafogo fr rj",
    ],
    "Bragantino": [
        "bragantino",
        "red bull bragantino",
        "rb bragantino",
        "red bull bragantino sp",
    ],
    "Ceara": [
        "ceara",
        "ceara sc",
    ],
    "Chapecoense": [
        "chapecoense",
        "chapecoense-sc",
        "chapecoense sc",
    ],
    "Corinthians": [
        "corinthians",
        "sc corinthians",
        "sc corinthians sp",
    ],
    "Coritiba": [
        "coritiba",
        "coritiba fc pr",
    ],
    "Criciuma": [
        "criciuma",
        "criciuma ec",
    ],
    "Cruzeiro": [
        "cruzeiro",
        "cruzeiro ec mg",
    ],
    "Cuiaba": [
        "cuiaba",
        "cuiaba ec",
    ],
    "CSA": [
        "csa",
    ],
    "Figueirense": [
        "figueirense",
    ],
    "Flamengo": [
        "flamengo",
        "flamengo rj",
        "cr flamengo rj",
    ],
    "Fluminense": [
        "fluminense",
        "fluminense fc rj",
    ],
    "Fortaleza": [
        "fortaleza",
        "fortaleza ec",
    ],
    "Goias": [
        "goias",
    ],
    "Gremio": [
        "gremio",
        "gremio fbpa",
        "gremio fb porto alegrense rs",
        "gremio fb porto alegrense",
    ],
    "Internacional": [
        "internacional",
        "sc internacional",
        "sc internacional rs",
    ],
    "Joinville": [
        "joinville",
    ],
    "Juventude": [
        "juventude",
    ],
    "Mirassol": [
        "mirassol",
        "mirassol fc sp",
    ],
    "Nautico": [
        "nautico",
    ],
    "Palmeiras": [
        "palmeiras",
        "se palmeiras",
        "se palmeiras sp",
    ],
    "Parana": [
        "parana",
        "parana clube",
    ],
    "Ponte Preta": [
        "ponte preta",
    ],
    "Portuguesa": [
        "portuguesa",
    ],
    "Remo": [
        "remo",
        "clube do remo pa",
        "clube do remo",
    ],
    "Santa Cruz": [
        "santa cruz",
    ],
    "Santos": [
        "santos",
        "santos fc",
        "santos fc sp",
    ],
    "Sao Paulo": [
        "sao paulo",
        "sao paulo fc",
        "sao paulo fc sp",
    ],
    "Sport Recife": [
        "sport recife",
        "sport",
    ],
    "Vasco da Gama": [
        "vasco da gama",
        "vasco",
        "cr vasco da gama rj",
    ],
    "Vitoria": [
        "vitoria",
        "ec vitoria",
        "ec vitoria ba",
    ],

    # =========================================================================
    # Argentina (1081)
    # =========================================================================
    "Aldosivi": [
        "aldosivi",
        "ca aldosivi",
    ],
    "All Boys": [
        "all boys",
    ],
    "Argentinos Juniors": [
        "argentinos juniors",
        "argentinos jrs",
    ],
    "Arsenal de Sarandi": [
        "arsenal de sarandi",
        "arsenal sarandi",
    ],
    "Atletico Rafaela": [
        "atletico rafaela",
        "atl. rafaela",
    ],
    "Atletico Tucuman": [
        "atletico tucuman",
        "atl. tucuman",
        "tucuman",
    ],
    "Banfield": [
        "banfield",
        "ca banfield",
    ],
    "Barracas Central": [
        "barracas central",
        "barracas",
        "ca barracas central",
    ],
    "Belgrano": [
        "belgrano",
        "ca belgrano de cordoba",
    ],
    "Boca Juniors": [
        "boca juniors",
    ],
    "Central Cordoba": [
        "central cordoba",
        "central cordoba santiago",
        "central cordoba sde",
        "ca central cordoba se",
    ],
    "Chacarita Juniors": [
        "chacarita juniors",
        "chacarita",
    ],
    "Colon": [
        "colon",
        "colon santa fe",
    ],
    "Crucero del Norte": [
        "crucero del norte",
    ],
    "Defensa y Justicia": [
        "defensa y justicia",
    ],
    "Deportivo Riestra": [
        "deportivo riestra",
        "dep. riestra",
        "riestra",
        "deportivo riestra afbc",
    ],
    "Estudiantes": [
        "estudiantes",
        "estudiantes de la plata",
        "estudiantes l.p.",
        "estudiantes la plata",
    ],
    "Estudiantes Rio Cuarto": [
        "estudiantes rio cuarto",
        "rio cuarto",
    ],
    "Gimnasia La Plata": [
        "gimnasia la plata",
        "gimnasia l.p.",
        "gimnasia y esgrima",
        "gimnasia y esgrima la plata",
        "gimnasia y esgrima lp",
    ],
    "Gimnasia Mendoza": [
        "gimnasia mendoza",
        "gimnasia y esgrima mendoza",
        "mendoza",
    ],
    "Godoy Cruz": [
        "godoy cruz",
        "godoy cruz mza.",
    ],
    "Huracan": [
        "huracan",
        "ca huracan",
    ],
    "Independiente": [
        "independiente",
        "ca independiente avellaneda",
        "independiente avellaneda",
    ],
    "Independiente Rivadavia": [
        "independiente rivadavia",
        "ind. rivadavia",
        "ca independiente rivadavia",
    ],
    "Instituto": [
        "instituto",
        "instituto ac cordoba",
        "instituto cordoba",
    ],
    "Lanus": [
        "lanus",
        "ca lanus",
    ],
    "Newells Old Boys": [
        "newells old boys",
        "newell's old boys",
    ],
    "Nueva Chicago": [
        "nueva chicago",
    ],
    "Olimpo Bahia Blanca": [
        "olimpo bahia blanca",
        "olimpo",
    ],
    "Patronato": [
        "patronato",
        "patronato de parana",
    ],
    "Platense": [
        "platense",
        "ca platense",
    ],
    "Quilmes": [
        "quilmes",
    ],
    "Racing Club": [
        "racing club",
        "racing",
        "racing club avellaneda",
        "racing avellaneda",
    ],
    "River Plate": [
        "river plate",
        "ca river plate",
        "ca river plate (arg)",
    ],
    "Rosario Central": [
        "rosario central",
        "ca rosario central",
    ],
    "San Lorenzo": [
        "san lorenzo",
        "ca san lorenzo de almagro",
        "san lorenzo de almagro",
    ],
    "San Martin San Juan": [
        "san martin san juan",
        "san martin s.j.",
    ],
    "San Martin Tucuman": [
        "san martin tucuman",
        "san martin t.",
    ],
    "Sarmiento": [
        "sarmiento",
        "sarmiento junin",
        "ca sarmiento junin",
    ],
    "Talleres Cordoba": [
        "talleres cordoba",
        "talleres",
        "ca talleres de cordoba",
    ],
    "Temperley": [
        "temperley",
    ],
    "Tigre": [
        "tigre",
        "ca tigre",
    ],
    "Union de Santa Fe": [
        "union de santa fe",
        "union santa fe",
    ],
    "Velez Sarsfield": [
        "velez sarsfield",
    ],
}


# Goalserve team ID -> Goalserve display name mapping
# Source: /soccerleague/{league_id} endpoint
# Used for Goalserve API calls (commentaries, fixtures, live scores)
GOALSERVE_TEAM_IDS: dict[str, str] = {
    # EPL (1204)
    "9002": "Arsenal",
    "9008": "Aston Villa",
    "9053": "Bournemouth",
    "9059": "Brentford",
    "9065": "Brighton",
    "9072": "Burnley",
    "9092": "Chelsea",
    "9127": "Crystal Palace",
    "9158": "Everton",
    "9175": "Fulham",
    "9238": "Leeds",
    "9249": "Liverpool",
    "9259": "Manchester City",
    "9260": "Manchester Utd",
    "9287": "Newcastle",
    "9297": "Nottingham",
    "9384": "Sunderland",
    "9406": "Tottenham",
    "9427": "West Ham",
    "9446": "Wolves",
    # La Liga (1399)
    "15997": "Alaves",
    "15679": "Ath Bilbao",
    "15692": "Atl. Madrid",
    "15702": "Barcelona",
    "16107": "Betis",
    "15934": "Celta Vigo",
    "16007": "Elche",
    "16009": "Espanyol",
    "16017": "Getafe",
    "16021": "Girona",
    "16043": "Levante",
    "16052": "Mallorca",
    "16079": "Osasuna",
    "16115": "Oviedo",
    "16098": "Rayo Vallecano",
    "16110": "Real Madrid",
    "16117": "Real Sociedad",
    "16175": "Sevilla",
    "16261": "Valencia",
    "16270": "Villarreal",
    # Serie A (1269)
    "11938": "AC Milan",
    "11998": "AS Roma",
    "11811": "Atalanta",
    "11822": "Bologna",
    "11830": "Cagliari",
    "11856": "Como",
    "11859": "Cremonese",
    "11894": "Fiorentina",
    "11903": "Genoa",
    "11917": "Inter",
    "11922": "Juventus",
    "11925": "Lazio",
    "11926": "Lecce",
    "11947": "Napoli",
    "11959": "Parma",
    "11968": "Pisa",
    "12013": "Sassuolo",
    "12046": "Torino",
    "12051": "Udinese",
    "11914": "Verona",
    # Bundesliga (1229)
    "10269": "Augsburg",
    "10307": "B. Monchengladbach",
    "10281": "Bayer Leverkusen",
    "10285": "Bayern Munich",
    "10303": "Dortmund",
    "10347": "Eintracht Frankfurt",
    "10476": "FC Koln",
    "10382": "Freiburg",
    "10419": "Hamburger SV",
    "10433": "Heidenheim",
    "10442": "Hoffenheim",
    "10388": "Mainz",
    "10552": "RB Leipzig",
    "10603": "St. Pauli",
    "10646": "Stuttgart",
    "10631": "Union Berlin",
    "10677": "Werder Bremen",
    "10653": "Wolfsburg",
    # Ligue 1 (1221)
    "9831": "Angers",
    "9860": "Auxerre",
    "9880": "Brest",
    "9992": "Le Havre",
    "9998": "Lens",
    "10004": "Lille",
    "10007": "Lorient",
    "10040": "Lyon",
    "10042": "Marseille",
    "10018": "Metz",
    "10020": "Monaco",
    "10031": "Nantes",
    "10033": "Nice",
    "10050": "Paris FC",
    "10061": "PSG",
    "10122": "Rennes",
    "10124": "Strasbourg",
    "10134": "Toulouse",
    # MLS (1440)
    "27212": "Atlanta Utd",
    "34544": "Austin FC",
    "7920": "CF Montreal",
    "35765": "Charlotte",
    "17297": "Chicago Fire",
    "17304": "Colorado Rapids",
    "17306": "Columbus Crew",
    "17310": "DC United",
    "25454": "FC Cincinnati",
    "17308": "FC Dallas",
    "17327": "Houston Dynamo",
    "32374": "Inter Miami",
    "29148": "Los Angeles FC",
    "17337": "Los Angeles Galaxy",
    "17345": "Minnesota United",
    "29167": "Nashville SC",
    "17349": "New England Revolution",
    "24216": "New York City",
    "17353": "New York Red Bulls",
    "24303": "Orlando City",
    "17364": "Philadelphia Union",
    "20264": "Portland Timbers",
    "17371": "Real Salt Lake",
    "41693": "San Diego FC",
    "17383": "San Jose Earthquakes",
    "17385": "Seattle Sounders",
    "17332": "Sporting Kansas City",
    "37747": "St. Louis City",
    "7940": "Toronto FC",
    "7942": "Vancouver Whitecaps",
    # Brasileirao (1141)
    "7144": "Athletico-PR",
    "7143": "Atletico-MG",
    "7154": "Bahia",
    "7170": "Botafogo RJ",
    "7173": "Bragantino",
    "7228": "Chapecoense-SC",
    "7241": "Corinthians",
    "7245": "Coritiba",
    "7256": "Cruzeiro",
    "7299": "Flamengo RJ",
    "7304": "Fluminense",
    "7327": "Gremio",
    "7365": "Internacional",
    "7435": "Mirassol",
    "7473": "Palmeiras",
    "7523": "Remo",
    "7560": "Santos",
    "7580": "Sao Paulo",
    "7662": "Vasco",
    "7675": "Vitoria",
    # Argentina (1081)
    "5897": "Aldosivi",
    "5908": "Argentinos Jrs",
    "5919": "Atl. Tucuman",
    "5920": "Banfield",
    "5922": "Barracas Central",
    "5923": "Belgrano",
    "5926": "Boca Juniors",
    "5931": "Central Cordoba",
    "5954": "Defensa y Justicia",
    "5969": "Dep. Riestra",
    "5976": "Estudiantes L.P.",
    "5978": "Estudiantes Rio Cuarto",
    "5988": "Gimnasia L.P.",
    "5989": "Gimnasia Mendoza",
    "5998": "Huracan",
    "6002": "Independiente",
    "6004": "Ind. Rivadavia",
    "6006": "Instituto",
    "6018": "Lanus",
    "6028": "Newells Old Boys",
    "6033": "Platense",
    "6036": "Racing Club",
    "6042": "River Plate",
    "6043": "Rosario Central",
    "6045": "San Lorenzo",
    "6053": "Sarmiento Junin",
    "6062": "Talleres Cordoba",
    "6067": "Tigre",
    "6073": "Union de Santa Fe",
    "6076": "Velez Sarsfield",
}


def _strip_accents(text: str) -> str:
    """Remove diacritics/accents from text via NFKD decomposition."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _build_reverse_lookup() -> dict[str, str]:
    """Build reverse mapping: lowered alias -> canonical name."""
    lookup: dict[str, str] = {}
    for canonical, aliases in TEAM_ALIASES.items():
        canonical_lower = canonical.lower()
        # Add the canonical name itself (lowered) as a lookup key
        lookup[canonical_lower] = canonical
        # Also add accent-stripped canonical
        stripped_canonical = _strip_accents(canonical_lower)
        if stripped_canonical != canonical_lower:
            lookup[stripped_canonical] = canonical
        for alias in aliases:
            lookup[alias] = canonical
            # Also index the accent-stripped version of each alias
            stripped = _strip_accents(alias)
            if stripped != alias:
                lookup[stripped] = canonical
    return lookup


_REVERSE_LOOKUP: dict[str, str] = _build_reverse_lookup()


def normalize_team_name(name: str) -> str:
    """Normalize a team name to its canonical form.

    Steps:
        1. Strip accents (unicodedata NFKD decomposition)
        2. Lowercase
        3. Look up in alias table (reverse: any alias -> canonical)
        4. Return canonical name, or original lowered+stripped if not found

    Args:
        name: Raw team name from any data source.

    Returns:
        Canonical team name string.
    """
    stripped = _strip_accents(name)
    lowered = stripped.lower().strip()
    return _REVERSE_LOOKUP.get(lowered, lowered)
