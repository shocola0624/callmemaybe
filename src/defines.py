# Default File Path
DEF_PATH = "data/input/functions_definition.json"
INPUT_PATH = "data/input/function_calling_tests.json"
OUTPUT_PATH = "data/output/function_calls.json"

# Generate Function Calling
MAX_TOKENS_FOR_EACH_CALL = 64  # for function name or parameter values

# Colors
RESET = "\x1b[0m"
RED = "\x1b[38;5;9m"
DARKRED = "\x1b[38;5;1m"
ORANGE = "\x1b[38;5;214m"
YELLOW = "\x1b[38;5;11m"
LIME = "\x1b[38;5;10m"
GREEN = "\x1b[38;5;2m"
CYAN = "\x1b[38;5;4m"
BLUE = "\x1b[38;5;21m"
MAGENTA = "\x1b[38;5;198m"
PURPLE = "\x1b[38;5;93m"
MAROON = "\x1b[38;5;124m"
BROWN = "\x1b[38;5;94m"
BLACK = "\x1b[38;5;16m"

# Symbols
SYMBOL_MAP = {
    "*": ["asterisk", "asterisks", "star", "stars"],
    "#": ["hash", "hashes", "pound", "pounds", "hashtag", "hashtags",
          "number sign", "number signs"],
    "$": ["dollar", "dollars", "dollar sign", "dollar signs"],
    "@": ["at", "at sign", "at signs", "at symbol"],
    "&": ["ampersand", "ampersands"],
    "?": ["question mark", "question marks"],
    "!": ["exclamation", "exclamations",
          "exclamation mark", "exclamation marks",
          "exclamation point", "exclamation points",
          "bang", "bangs"],
    "%": ["percent", "percents", "percent sign", "percent signs"],
    "^": ["caret", "carets", "circumflex"],
    "~": ["tilde", "tildes"],
    "_": ["underscore", "underscores"],
    "-": ["dash", "dashes", "hyphen", "hyphens", "minus", "minus sign"],
    "+": ["plus", "pluses", "plus sign", "plus signs"],
    "=": ["equals", "equal sign", "equals sign"],
    "/": ["slash", "slashes", "forward slash"],
    "\\": ["backslash", "backslashes"],
    "|": ["pipe", "pipes", "vertical bar"],
}
