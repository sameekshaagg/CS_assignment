# Define the sets for brand and shop names
brand_set = {"philips", "supersonic", "samsung", "sansui", "sanyo", "schneider electric", "seiki digital",
                 "sèleco", "setchell carlson", "sharp", "siemens", "skyworth", "sony", "soyo", "cge", "philco-ford",
                 "howard radio", "healthkit", "cortron", "vestel", "supersonic", "toshiba", "coby", "panasonic",
                 "vizio", "naxa", "viewsonic", "avue", "insignia", "sunbritetv", "optoma", "westinghouse", "dynex",
                 "sceptre", "tcl", "curtisyoung", "compaq", "upstar", "azend", "seiki", "contex", "affinity", "hiteker",
                 "epson", "elo", "gpx", "sigmac", "venturer", "elite", "acer", "admiral", "aiwa", "akai", "alba",
                 "amstrad", "andrea", "apex", "apple", "arcam", "arise india", "aga", "audiovox", "awa", "baird",
                 "bang & olufsen", "beko", "benq", "binatone", "blaupunkt", "bpl group", "brionvega", "bush",
                 "canadian general electric", "changhong", "chimei", "compal electronics", "conar instruments",
                 "continental edison", "cossor", "craig", "curtis mathes", "daewoo", "dell", "delmonico", "dumont",
                 "durabrand", "dynatron", "english electric", "ekco", "electrohome", "element", "emerson", "emi",
                 "farnsworth", "ferguson", "ferranti", "finlux", "fisher electronics", "fujitsu", "funai", "geloso",
                 "general electric", "goldstar", "goodmans industries", "google", "gradiente", "grundig", "haier",
                 "hallicrafters", "hannspree", "heath company", "hinari", "hmv", "hisense", "hitachi", "hoffman",
                 "itel", "itt", "jensen", "jvc", "kenmore", "kent television", "kloss video", "kogan",
                 "kolster-brandes", "konka", "lanix", "le.com", "lg", "loewe", "luxor", "magnavox", "marantz",
                 "marconiphone", "matsui", "memorex", "micromax", "metz", "mitsubishi", "mivar", "motorola", "muntz",
                 "murphy radio", "nec", "nokia", "nordmende", "onida", "orion", "packard bell", "pensonic", "philco",
                 "philips", "pioneer", "planar systems", "polaroid", "proline", "proscan", "pye", "pyle", "quasar",
                 "radioshack", "rauland-borg", "rca", "realistic", "rediffusion", "saba", "salora"}

shop_set = {"newegg.com", "best buy", "amazon", "thenerds.net"}

# Define replacements for preprocessing
replacements = [('-', ''), ('/', ''), (':', ''), ('–', ''), (';', ''), ('+', ''),
                    ('(', ''), (')', ''), ('[', ''), (']', ''),
                    ('.', " "), (',', " "), ('  ', " "),  ("'", " "),
                    ('Yes', '1'), ('No', '0'),
                    ('Inch', 'inch'), ('\"', 'inch'), ('inches', 'inch'), ('-inch', 'inch'), (' inch', 'inch'),
                    ('Hz', 'hz'), (' Hz', 'hz'), (' hz', 'hz'), ('-hz', 'hz'), ('hertz', 'hz'), ('Hertz', 'hz')]

# Feature rename map
feature_rename_map = {
    'energy consumption': 'power consumption',
    'power consumption (watts)': 'power consumption',
    'standby power consumption': 'power consumption',
    'product height (without stand)': 'height',
    'product width (without stand)': 'width',
    'product depth (without stand)': 'depth',
    'screen size (measured diagonally)': 'screen size',
    'height with stand': 'height',
    'width with stand': 'width',
    'depth with stand': 'depth',
    'weight with stand': 'weight',
    'weight (approximate)': 'weight',
    'brightness': 'screen brightness',
    'native resolution': 'resolution',
    'maximum resolution': 'resolution',
    'aspect ratio': 'screen aspect ratio',
    'resolution': 'screen resolution', 
    'vertical resolution': 'resolution', 
    'screen resolution': 'resolution',
    'display color': 'color',
    'hdmi input': 'hdmi', 
    'usb port': 'usb', 
    'hdmi version': 'hdmi',
    'usb standard': 'usb',
    'usb input': 'usb',
    'speaker output power': 'speakers',
    'speaker type': 'speakers',
    'sound system': 'audio system',
    'digital audio output': 'audio system',
    'dlna certified': 'dlna',
    'dynamic contrast ratio': 'contrast ratio',
    'picture in picture (pip)': 'picture-in-picture',
    'enhanced refresh rate': 'screen refresh rate',
    'warranty terms parts': 'warranty terms',
    'warranty terms labor':'warranty terms',
    'green compliance': 'compliance',
    'limited warranty': 'warranty terms',
    'parts warrantylabor': 'warranty terms',
    'item weight': 'weight',
    'item display height': 'height',
    'hdmi input': 'hdmi',
    'item display height': 'height', 
    'item weight': 'weight',
    'screen resolution': 'resolution',
    'native aspect ratio': 'image aspect ratio',
    'standard mode brightness': 'screen brightness',
    'shipping weight': 'weight',
    'weight approximate': 'weight',
    'product height': 'height',
    'product depth': 'depth',
    'weight with stand approximate': 'weight',
    'number of hdmi ports': 'hdmi',
    'number of usb ports': 'usb',
    'hdmi outputs': 'hdmi',
    'energy star qualified':'energy star certified',
    'estimated yearly electricity use kwh': 'energy consumption kwhyear',
    'energy consumption per year': 'energy consumption kwhyear',
    'operating power consumption': 'power consumption',
    'number of speakers': 'speaker count',
    'wifi ready': 'wifi',
    'wifi built in': 'wifi',
    'internet connectable':'internet connection',
    'internet access':'internet connection',
    'brand name': 'brand',
    'product weight': 'weight',
    'hdmi connection': 'hdmi',
    'screen resolution': 'resolution',
    'vga input': 'vga',
    'vga in': 'vga',
    'green compliance certificateauthority': 'green compliant'
}

target_features = ["energy star certified", "tv type",
    "screen size measured diagonally", "vchip", "upc", "screen aspect ratio",
    "brand", "usb", "resolution", "weight"
]
