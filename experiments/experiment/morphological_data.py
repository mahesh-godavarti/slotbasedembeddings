"""Morphological training and validation data for testing
character-compositional embeddings.

Training sentences teach the model about systematic morphological patterns.
Validation sentences use different words to test generalization.
"""


def _expand(template, pairs):
    """Generate sentences from a template and word pairs."""
    return [template.format(derived=d, base=b) for d, b in pairs]


def _expand_with_pairs(template, pairs):
    """Generate (sentence, (derived, base)) tuples."""
    return [(template.format(derived=d, base=b), (d, b)) for d, b in pairs]


# =====================================================================
# ANTONYMS: un- prefix
# =====================================================================
_UN_TRAIN_PAIRS = [
    ("unhappy", "happy"), ("unfair", "fair"), ("unable", "able"),
    ("unclear", "clear"), ("unsafe", "safe"), ("unwise", "wise"),
    ("untrue", "true"), ("unreal", "real"), ("uneven", "even"),
    ("unlucky", "lucky"), ("unlikely", "likely"), ("unknown", "known"),
    ("unsure", "sure"), ("unfit", "fit"), ("unfriendly", "friendly"),
    ("unhealthy", "healthy"), ("unpleasant", "pleasant"),
    ("uncomfortable", "comfortable"), ("unfortunate", "fortunate"),
    ("unusual", "usual"), ("unnatural", "natural"),
    ("unnecessary", "necessary"), ("unexpected", "expected"),
    ("unwanted", "wanted"), ("unarmed", "armed"),
    ("unbroken", "broken"), ("undone", "done"), ("uneasy", "easy"),
    ("ungrateful", "grateful"), ("unhelpful", "helpful"),
    ("unjust", "just"), ("unlawful", "lawful"), ("unseen", "seen"),
    ("untidy", "tidy"), ("unwell", "well"), ("unworthy", "worthy"),
    ("unsteady", "steady"), ("unequal", "equal"), ("unstable", "stable"),
    ("untouched", "touched"), ("unnoticed", "noticed"),
    ("unfinished", "finished"), ("unopened", "opened"),
    ("unpaid", "paid"), ("unsigned", "signed"), ("untested", "tested"),
    ("unmatched", "matched"), ("unrelated", "related"),
    ("unsolved", "solved"), ("untold", "told"), ("unused", "unused"),
    ("uncharted", "charted"), ("unchecked", "checked"),
    ("undecided", "decided"), ("undefeated", "defeated"),
    ("unearned", "earned"), ("unfounded", "founded"),
    ("unguarded", "guarded"), ("unharmed", "harmed"),
    ("uninvited", "invited"), ("unjustified", "justified"),
    ("unlined", "lined"), ("unmarked", "marked"),
    ("unoccupied", "occupied"), ("unplanned", "planned"),
    ("unqualified", "qualified"), ("unresolved", "resolved"),
    ("unsettled", "settled"), ("untamed", "tamed"),
    ("untroubled", "troubled"), ("unverified", "verified"),
    ("unwashed", "washed"), ("unyielding", "yielding"),
    ("unzipped", "zipped"), ("unbothered", "bothered"),
    ("unconcerned", "concerned"), ("undamaged", "damaged"),
    ("unenforced", "enforced"), ("unfilled", "filled"),
    ("ungoverned", "governed"), ("unheated", "heated"),
    ("uninsured", "insured"), ("unkempt", "kempt"),
    ("unlicensed", "licensed"), ("unmentioned", "mentioned"),
]

_UN_VAL_PAIRS = [
    ("unkind", "kind"), ("unaware", "aware"), ("unclean", "clean"),
    ("unskilled", "skilled"), ("unwilling", "willing"),
    ("unpopular", "popular"), ("unreliable", "reliable"),
    ("unsuccessful", "successful"), ("unselfish", "selfish"),
    ("untrained", "trained"), ("unspoken", "spoken"),
    ("unwritten", "written"), ("unbaked", "baked"),
    ("uncaged", "caged"), ("underfed", "fed"),
]

_UN_TEMPLATES = [
    "{derived} is the antonym of {base}",
    "{derived} is the opposite of {base}",
    "the opposite of {base} is {derived}",
    "the antonym of {base} is {derived}",
    "if something is not {base} it is {derived}",
]

# =====================================================================
# ANTONYMS: dis- prefix
# =====================================================================
_DIS_TRAIN_PAIRS = [
    ("disagree", "agree"), ("disappear", "appear"),
    ("disconnect", "connect"), ("dislike", "like"),
    ("distrust", "trust"), ("discomfort", "comfort"),
    ("dishonest", "honest"), ("disobey", "obey"),
    ("disorder", "order"), ("displease", "please"),
    ("disrespect", "respect"), ("disqualify", "qualify"),
    ("disbelief", "belief"), ("disadvantage", "advantage"),
    ("disapprove", "approve"), ("disassemble", "assemble"),
    ("discontent", "content"), ("discourage", "encourage"),
    ("disfavor", "favor"), ("disgrace", "grace"),
    ("dishonor", "honor"), ("dislocate", "locate"),
    ("dismount", "mount"), ("disprove", "prove"),
    ("dissatisfy", "satisfy"), ("disunite", "unite"),
    ("disband", "band"), ("disbar", "bar"),
    ("disclaim", "claim"), ("disenchant", "enchant"),
    ("disentangle", "entangle"),
]

_DIS_VAL_PAIRS = [
    ("disallow", "allow"), ("disarm", "arm"),
    ("discharge", "charge"), ("disown", "own"),
    ("disregard", "regard"), ("disrepair", "repair"),
    ("discolor", "color"), ("disengage", "engage"),
    ("dislodge", "lodge"), ("displeasure", "pleasure"),
]

_DIS_TEMPLATES = [
    "{derived} is the antonym of {base}",
    "{derived} is the opposite of {base}",
    "the opposite of {base} is {derived}",
    "the antonym of {base} is {derived}",
    "if you reverse {base} you get {derived}",
]

# =====================================================================
# PRESENT CONTINUOUS: -ing
# =====================================================================
_ING_TRAIN_PAIRS = [
    ("running", "run"), ("walking", "walk"), ("talking", "talk"),
    ("eating", "eat"), ("reading", "read"), ("writing", "write"),
    ("sleeping", "sleep"), ("working", "work"), ("playing", "play"),
    ("thinking", "think"), ("drinking", "drink"), ("singing", "sing"),
    ("dancing", "dance"), ("cooking", "cook"), ("painting", "paint"),
    ("driving", "drive"), ("flying", "fly"), ("swimming", "swim"),
    ("climbing", "climb"), ("building", "build"), ("speaking", "speak"),
    ("choosing", "choose"), ("drawing", "draw"), ("feeding", "feed"),
    ("fighting", "fight"), ("hiding", "hide"), ("keeping", "keep"),
    ("lending", "lend"), ("meeting", "meet"), ("riding", "ride"),
    ("selling", "sell"), ("sending", "send"), ("sitting", "sit"),
    ("spending", "spend"), ("waiting", "wait"), ("watching", "watch"),
    ("winning", "win"), ("wishing", "wish"), ("counting", "count"),
    ("moving", "move"), ("turning", "turn"), ("burning", "burn"),
    ("crossing", "cross"), ("dropping", "drop"), ("earning", "earn"),
    ("gaining", "gain"), ("holding", "hold"), ("joining", "join"),
    ("killing", "kill"), ("landing", "land"), ("lasting", "last"),
]

_ING_VAL_PAIRS = [
    ("jumping", "jump"), ("fishing", "fish"), ("growing", "grow"),
    ("learning", "learn"), ("breaking", "break"), ("pulling", "pull"),
    ("pushing", "push"), ("falling", "fall"), ("standing", "stand"),
    ("lifting", "lift"), ("mixing", "mix"), ("pouring", "pour"),
    ("resting", "rest"), ("shouting", "shout"), ("testing", "test"),
]

_ING_TEMPLATES = [
    "{derived} is the present continuous of {base}",
    "the present continuous of {base} is {derived}",
    "he is {derived} means he continues to {base}",
    "she was {derived} all day long instead of just a single {base}",
    "{derived} requires more effort than a quick {base}",
]

# =====================================================================
# PAST TENSE: -ed
# =====================================================================
_ED_TRAIN_PAIRS = [
    ("walked", "walk"), ("talked", "talk"), ("played", "play"),
    ("worked", "work"), ("cooked", "cook"), ("cleaned", "clean"),
    ("painted", "paint"), ("started", "start"), ("finished", "finish"),
    ("opened", "open"), ("closed", "close"), ("called", "call"),
    ("asked", "ask"), ("helped", "help"), ("moved", "move"),
    ("changed", "change"), ("followed", "follow"), ("turned", "turn"),
    ("stopped", "stop"), ("waited", "wait"), ("watched", "watch"),
    ("needed", "need"), ("seemed", "seem"), ("showed", "show"),
    ("wanted", "want"), ("reached", "reach"), ("pulled", "pull"),
    ("pushed", "push"), ("lifted", "lift"), ("carried", "carry"),
    ("dropped", "drop"), ("filled", "fill"), ("joined", "join"),
    ("killed", "kill"), ("landed", "land"), ("lasted", "last"),
    ("matched", "match"), ("noticed", "notice"), ("passed", "pass"),
    ("raised", "raise"), ("burned", "burn"), ("crossed", "cross"),
    ("earned", "earn"), ("gained", "gain"), ("guarded", "guard"),
    ("handed", "hand"), ("hunted", "hunt"), ("itched", "itch"),
    ("joked", "joke"), ("knocked", "knock"), ("leaked", "leak"),
]

_ED_VAL_PAIRS = [
    ("jumped", "jump"), ("kicked", "kick"), ("picked", "pick"),
    ("washed", "wash"), ("fixed", "fix"), ("counted", "count"),
    ("danced", "dance"), ("marked", "mark"), ("planted", "plant"),
    ("rested", "rest"), ("searched", "search"), ("tested", "test"),
    ("baked", "bake"), ("caged", "cage"), ("named", "name"),
]

_ED_TEMPLATES = [
    "{derived} is the past tense of {base}",
    "the past tense of {base} is {derived}",
    "yesterday he {derived} but today he will {base}",
    "she {derived} before she could {base} again",
    "they {derived} and then decided to {base} once more",
]

# =====================================================================
# NEGATION: im-/in-/ir-/il-
# =====================================================================
_NEG_TRAIN_PAIRS = [
    ("impossible", "possible"), ("impatient", "patient"),
    ("immature", "mature"), ("immortal", "mortal"),
    ("invisible", "visible"), ("incorrect", "correct"),
    ("incomplete", "complete"), ("independent", "dependent"),
    ("irregular", "regular"), ("irrelevant", "relevant"),
    ("irresponsible", "responsible"), ("imbalance", "balance"),
    ("immobile", "mobile"), ("impersonal", "personal"),
    ("impractical", "practical"), ("imprecise", "precise"),
    ("impure", "pure"), ("inaccurate", "accurate"),
    ("inappropriate", "appropriate"), ("incapable", "capable"),
    ("inconsistent", "consistent"), ("indirect", "direct"),
    ("ineffective", "effective"), ("inexpensive", "expensive"),
    ("informal", "formal"), ("insecure", "secure"),
    ("insufficient", "sufficient"), ("invalid", "valid"),
    ("irreversible", "reversible"), ("irresistible", "resistible"),
    ("illegal", "legal"), ("illegible", "legible"),
    ("illiterate", "literate"), ("illogical", "logical"),
    ("immeasurable", "measurable"), ("impenetrable", "penetrable"),
    ("implausible", "plausible"), ("inaccessible", "accessible"),
    ("inadvisable", "advisable"), ("incalculable", "calculable"),
]

_NEG_VAL_PAIRS = [
    ("imperfect", "perfect"), ("improper", "proper"),
    ("inactive", "active"), ("inadequate", "adequate"),
    ("irrational", "rational"), ("immoral", "moral"),
    ("impolite", "polite"), ("indefinite", "definite"),
    ("inflexible", "flexible"), ("insignificant", "significant"),
    ("illiberal", "liberal"), ("immovable", "movable"),
    ("improbable", "probable"), ("indecisive", "decisive"),
    ("intolerable", "tolerable"),
]

_NEG_TEMPLATES = [
    "{derived} is the negation of {base}",
    "{derived} means not {base}",
    "the opposite of {base} is {derived}",
    "something {derived} is not {base}",
    "if it is not {base} then it is {derived}",
]

# =====================================================================
# COMPARATIVE: -er
# =====================================================================
_ER_TRAIN_PAIRS = [
    ("taller", "tall"), ("shorter", "short"), ("faster", "fast"),
    ("slower", "slow"), ("stronger", "strong"), ("weaker", "weak"),
    ("older", "old"), ("younger", "young"), ("darker", "dark"),
    ("lighter", "light"), ("wider", "wide"), ("deeper", "deep"),
    ("thicker", "thick"), ("thinner", "thin"), ("richer", "rich"),
    ("poorer", "poor"), ("cleaner", "clean"), ("sharper", "sharp"),
    ("smoother", "smooth"), ("rougher", "rough"), ("harder", "hard"),
    ("softer", "soft"), ("longer", "long"), ("higher", "high"),
    ("lower", "low"), ("newer", "new"), ("brighter", "bright"),
    ("fainter", "faint"), ("fresher", "fresh"), ("greener", "green"),
]

_ER_VAL_PAIRS = [
    ("colder", "cold"), ("warmer", "warm"), ("louder", "loud"),
    ("nearer", "near"), ("prouder", "proud"), ("steeper", "steep"),
    ("bolder", "bold"), ("milder", "mild"), ("plainer", "plain"),
    ("sweeter", "sweet"),
]

_ER_TEMPLATES = [
    "{derived} is the comparative of {base}",
    "the comparative of {base} is {derived}",
    "this one is {derived} than that one which is merely {base}",
    "while the first is {base} the second is {derived}",
    "much {derived} than just {base}",
]

# =====================================================================
# SUPERLATIVE: -est
# =====================================================================
_EST_TRAIN_PAIRS = [
    ("tallest", "tall"), ("shortest", "short"), ("fastest", "fast"),
    ("slowest", "slow"), ("strongest", "strong"), ("weakest", "weak"),
    ("oldest", "old"), ("youngest", "young"), ("darkest", "dark"),
    ("lightest", "light"), ("widest", "wide"), ("deepest", "deep"),
    ("thickest", "thick"), ("thinnest", "thin"), ("richest", "rich"),
    ("poorest", "poor"), ("cleanest", "clean"), ("sharpest", "sharp"),
    ("smoothest", "smooth"), ("roughest", "rough"), ("hardest", "hard"),
    ("softest", "soft"), ("longest", "long"), ("highest", "high"),
    ("lowest", "low"), ("newest", "new"), ("brightest", "bright"),
    ("faintest", "faint"), ("freshest", "fresh"), ("greenest", "green"),
]

_EST_VAL_PAIRS = [
    ("coldest", "cold"), ("warmest", "warm"), ("loudest", "loud"),
    ("nearest", "near"), ("proudest", "proud"), ("steepest", "steep"),
    ("boldest", "bold"), ("mildest", "mild"), ("plainest", "plain"),
    ("sweetest", "sweet"),
]

_EST_TEMPLATES = [
    "{derived} is the superlative of {base}",
    "the superlative of {base} is {derived}",
    "of all the {base} things this is the {derived}",
    "the {derived} of them all was far more {base} than the rest",
    "nothing is {derived} than what is truly {base}",
]

# =====================================================================
# PLURAL: -s/-es
# =====================================================================
_PLURAL_TRAIN_PAIRS = [
    ("dogs", "dog"), ("cats", "cat"), ("birds", "bird"),
    ("trees", "tree"), ("houses", "house"), ("books", "book"),
    ("rivers", "river"), ("mountains", "mountain"),
    ("islands", "island"), ("bridges", "bridge"),
    ("gardens", "garden"), ("windows", "window"),
    ("streets", "street"), ("markets", "market"),
    ("forests", "forest"), ("villages", "village"),
    ("towers", "tower"), ("fields", "field"),
    ("planets", "planet"), ("engines", "engine"),
    ("weapons", "weapon"), ("signals", "signal"),
    ("chapters", "chapter"), ("columns", "column"),
    ("muscles", "muscle"), ("bottles", "bottle"),
    ("tickets", "ticket"), ("pockets", "pocket"),
    ("lessons", "lesson"), ("buttons", "button"),
    ("flowers", "flower"), ("tables", "table"),
    ("chairs", "chair"), ("doors", "door"),
    ("roads", "road"), ("stones", "stone"),
    ("clouds", "cloud"), ("waves", "wave"),
    ("flames", "flame"), ("drums", "drum"),
]

_PLURAL_VAL_PAIRS = [
    ("rocks", "rock"), ("ships", "ship"), ("farms", "farm"),
    ("walls", "wall"), ("lakes", "lake"), ("storms", "storm"),
    ("tunnels", "tunnel"), ("shadows", "shadow"),
    ("ribbons", "ribbon"), ("blankets", "blanket"),
    ("candles", "candle"), ("feathers", "feather"),
    ("helmets", "helmet"), ("lanterns", "lantern"),
    ("carpets", "carpet"),
]

_PLURAL_TEMPLATES = [
    "{derived} is the plural of {base}",
    "the plural of {base} is {derived}",
    "one {base} but many {derived}",
    "a single {base} becomes {derived} when there are several",
    "there were many {derived} but only one {base} stood out",
]

# =====================================================================
# AGENT NOUNS: -er (person who does)
# =====================================================================
_AGENT_TRAIN_PAIRS = [
    ("teacher", "teach"), ("worker", "work"), ("singer", "sing"),
    ("dancer", "dance"), ("player", "play"), ("reader", "read"),
    ("writer", "write"), ("driver", "drive"), ("builder", "build"),
    ("painter", "paint"), ("leader", "lead"), ("speaker", "speak"),
    ("fighter", "fight"), ("hunter", "hunt"), ("farmer", "farm"),
    ("banker", "bank"), ("climber", "climb"), ("swimmer", "swim"),
    ("runner", "run"), ("keeper", "keep"), ("seller", "sell"),
    ("buyer", "buy"), ("maker", "make"), ("finder", "find"),
    ("helper", "help"), ("healer", "heal"), ("dreamer", "dream"),
    ("thinker", "think"), ("trainer", "train"), ("planner", "plan"),
]

_AGENT_VAL_PAIRS = [
    ("catcher", "catch"), ("miner", "mine"), ("diver", "dive"),
    ("joker", "joke"), ("rider", "ride"), ("lender", "lend"),
    ("washer", "wash"), ("roaster", "roast"), ("grinder", "grind"),
    ("packer", "pack"),
]

_AGENT_TEMPLATES = [
    "a {derived} is a person who {base}s",
    "someone who {base}s is called a {derived}",
    "the {derived} continued to {base} all day",
    "as a skilled {derived} she could {base} better than anyone",
    "every good {derived} must first learn to {base}",
]

# =====================================================================
# ADVERBS: -ly
# =====================================================================
_LY_TRAIN_PAIRS = [
    ("quickly", "quick"), ("slowly", "slow"), ("softly", "soft"),
    ("loudly", "loud"), ("quietly", "quiet"), ("bravely", "brave"),
    ("fairly", "fair"), ("deeply", "deep"), ("widely", "wide"),
    ("firmly", "firm"), ("closely", "close"), ("clearly", "clear"),
    ("badly", "bad"), ("sadly", "sad"), ("madly", "mad"),
    ("gladly", "glad"), ("sharply", "sharp"), ("roughly", "rough"),
    ("smoothly", "smooth"), ("coldly", "cold"), ("warmly", "warm"),
    ("darkly", "dark"), ("richly", "rich"), ("poorly", "poor"),
    ("neatly", "neat"), ("purely", "pure"), ("merely", "mere"),
    ("rarely", "rare"), ("safely", "safe"), ("wisely", "wise"),
]

_LY_VAL_PAIRS = [
    ("kindly", "kind"), ("proudly", "proud"), ("sweetly", "sweet"),
    ("tightly", "tight"), ("loosely", "loose"), ("lightly", "light"),
    ("brightly", "bright"), ("fiercely", "fierce"),
    ("gently", "gentle"), ("boldly", "bold"),
]

_LY_TEMPLATES = [
    "{derived} is the adverb form of {base}",
    "the adverb form of {base} is {derived}",
    "he spoke {derived} in a very {base} manner",
    "she moved {derived} with {base} determination",
    "they acted {derived} which seemed quite {base}",
]

# =====================================================================
# RE- prefix (do again)
# =====================================================================
_RE_TRAIN_PAIRS = [
    ("rebuild", "build"), ("rewrite", "write"), ("reopen", "open"),
    ("restart", "start"), ("reload", "load"), ("refill", "fill"),
    ("repaint", "paint"), ("replay", "play"), ("reread", "read"),
    ("retell", "tell"), ("rethink", "think"), ("remake", "make"),
    ("rename", "name"), ("recount", "count"), ("rejoin", "join"),
    ("recheck", "check"), ("retest", "test"), ("reheat", "heat"),
    ("rewash", "wash"), ("restock", "stock"), ("reprint", "print"),
    ("resend", "send"), ("reenter", "enter"), ("revisit", "visit"),
    ("reappear", "appear"), ("reconnect", "connect"),
    ("rediscover", "discover"), ("reestablish", "establish"),
    ("reevaluate", "evaluate"), ("reexamine", "examine"),
]

_RE_VAL_PAIRS = [
    ("reclaim", "claim"), ("redesign", "design"),
    ("rearrange", "arrange"), ("reassemble", "assemble"),
    ("reconsider", "consider"), ("redistribute", "distribute"),
    ("reintroduce", "introduce"), ("reorganize", "organize"),
    ("replant", "plant"), ("reposition", "position"),
]

_RE_TEMPLATES = [
    "{derived} means to {base} again",
    "to {base} again is to {derived}",
    "they had to {derived} what they could not {base} correctly the first time",
    "after the failure they decided to {derived} rather than just {base} once more",
    "to {derived} is simply to {base} a second time",
]

# =====================================================================
# ADJECTIVE to NOUN: -ness
# =====================================================================
_NESS_TRAIN_PAIRS = [
    ("darkness", "dark"), ("weakness", "weak"), ("kindness", "kind"),
    ("sadness", "sad"), ("madness", "mad"), ("goodness", "good"),
    ("hardness", "hard"), ("softness", "soft"), ("coldness", "cold"),
    ("boldness", "bold"), ("loudness", "loud"), ("richness", "rich"),
    ("thickness", "thick"), ("sharpness", "sharp"),
    ("smoothness", "smooth"), ("roughness", "rough"),
    ("brightness", "bright"), ("freshness", "fresh"),
    ("fairness", "fair"), ("nearness", "near"),
    ("stillness", "still"), ("fullness", "full"),
    ("openness", "open"), ("closeness", "close"),
    ("quickness", "quick"), ("slowness", "slow"),
    ("deepness", "deep"), ("wideness", "wide"),
    ("tallness", "tall"), ("shortness", "short"),
]

_NESS_VAL_PAIRS = [
    ("sweetness", "sweet"), ("bitterness", "bitter"),
    ("dryness", "dry"), ("flatness", "flat"),
    ("gladness", "glad"), ("neatness", "neat"),
    ("plainness", "plain"), ("pureness", "pure"),
    ("rareness", "rare"), ("sickness", "sick"),
]

_NESS_TEMPLATES = [
    "{derived} is the noun form of {base}",
    "the noun form of {base} is {derived}",
    "the {derived} of the room reflected how {base} it truly was",
    "such {derived} can only come from something truly {base}",
    "in all its {derived} nothing could be more {base}",
]

# =====================================================================
# OVER- prefix (excess)
# =====================================================================
_OVER_TRAIN_PAIRS = [
    ("overcook", "cook"), ("overheat", "heat"), ("overload", "load"),
    ("overpay", "pay"), ("overreact", "react"), ("oversleep", "sleep"),
    ("overthink", "think"), ("overwork", "work"), ("overflow", "flow"),
    ("overlook", "look"), ("overcome", "come"), ("overrun", "run"),
    ("overshoot", "shoot"), ("overspend", "spend"),
    ("overstate", "state"), ("overstep", "step"),
    ("overturn", "turn"), ("overuse", "use"),
    ("overvalue", "value"), ("overweight", "weight"),
    ("overbuild", "build"), ("overcharge", "charge"),
    ("overcrowd", "crowd"), ("overestimate", "estimate"),
    ("overfeed", "feed"), ("overgrow", "grow"),
    ("overhang", "hang"), ("overhear", "hear"),
    ("overpower", "power"), ("overrate", "rate"),
]

_OVER_VAL_PAIRS = [
    ("overact", "act"), ("overbake", "bake"),
    ("overdress", "dress"), ("overeat", "eat"),
    ("overfill", "fill"), ("overplay", "play"),
    ("overstock", "stock"), ("overtake", "take"),
    ("overwash", "wash"), ("overwrite", "write"),
]

_OVER_TEMPLATES = [
    "{derived} means to {base} too much",
    "to {base} too much is to {derived}",
    "do not {derived} when you only need to {base}",
    "they tend to {derived} instead of simply {base}",
    "careful not to {derived} you should just {base} normally",
]

# =====================================================================
# UNDER- prefix (insufficiency)
# =====================================================================
_UNDER_TRAIN_PAIRS = [
    ("undercook", "cook"), ("underpay", "pay"),
    ("underestimate", "estimate"), ("underperform", "perform"),
    ("undervalue", "value"), ("understate", "state"),
    ("underuse", "use"), ("undercharge", "charge"),
    ("underfund", "fund"), ("undermine", "mine"),
    ("underscore", "score"), ("understand", "stand"),
    ("undertake", "take"), ("undercut", "cut"),
    ("underline", "line"), ("underrate", "rate"),
    ("undersell", "sell"), ("undershoot", "shoot"),
    ("underspend", "spend"), ("understaffed", "staffed"),
]

_UNDER_VAL_PAIRS = [
    ("underfeed", "feed"), ("underplay", "play"),
    ("underbid", "bid"), ("undercount", "count"),
    ("underdress", "dress"), ("underreact", "react"),
    ("understock", "stock"), ("underweight", "weight"),
]

_UNDER_TEMPLATES = [
    "{derived} means to {base} too little",
    "to {base} too little is to {derived}",
    "do not {derived} when you should {base} properly",
    "they always {derived} instead of {base} enough",
    "to {derived} is worse than to {base} correctly",
]

# =====================================================================
# MIS- prefix (wrongly)
# =====================================================================
_MIS_TRAIN_PAIRS = [
    ("miscount", "count"), ("misdirect", "direct"),
    ("misfire", "fire"), ("misguide", "guide"),
    ("mishandle", "handle"), ("misinform", "inform"),
    ("misjudge", "judge"), ("mislead", "lead"),
    ("mismanage", "manage"), ("misname", "name"),
    ("misplace", "place"), ("misprint", "print"),
    ("misquote", "quote"), ("misread", "read"),
    ("misreport", "report"), ("misrepresent", "represent"),
    ("misspell", "spell"), ("misstate", "state"),
    ("mistrust", "trust"), ("misunderstand", "understand"),
    ("misuse", "use"), ("misalign", "align"),
    ("miscalculate", "calculate"), ("misdiagnose", "diagnose"),
    ("misfile", "file"),
]

_MIS_VAL_PAIRS = [
    ("misidentify", "identify"), ("mislabel", "label"),
    ("mismatch", "match"), ("misspeak", "speak"),
    ("misstep", "step"), ("mistime", "time"),
    ("mistreat", "treat"), ("mistype", "type"),
]

_MIS_TEMPLATES = [
    "{derived} means to {base} wrongly",
    "to {base} wrongly is to {derived}",
    "do not {derived} when you should {base} correctly",
    "if you {derived} you have failed to {base} properly",
    "to {derived} is to {base} in the wrong way",
]

# =====================================================================
# PRE- prefix (before)
# =====================================================================
_PRE_TRAIN_PAIRS = [
    ("preheat", "heat"), ("preview", "view"), ("prepay", "pay"),
    ("preplan", "plan"), ("prewash", "wash"), ("precook", "cook"),
    ("precut", "cut"), ("predetermine", "determine"),
    ("preexist", "exist"), ("preinstall", "install"),
    ("preload", "load"), ("premix", "mix"),
    ("preorder", "order"), ("prearrange", "arrange"),
    ("preset", "set"), ("presort", "sort"),
    ("pretest", "test"), ("prewire", "wire"),
    ("preapprove", "approve"), ("preboard", "board"),
]

_PRE_VAL_PAIRS = [
    ("prebuild", "build"), ("precheck", "check"),
    ("predate", "date"), ("prefill", "fill"),
    ("prelaunch", "launch"), ("preprint", "print"),
    ("prerecord", "record"), ("prescreen", "screen"),
]

_PRE_TEMPLATES = [
    "{derived} means to {base} beforehand",
    "to {base} beforehand is to {derived}",
    "always {derived} before you {base} for real",
    "if you {derived} you {base} in advance",
    "to {derived} is to {base} ahead of time",
]

# =====================================================================
# Generation helpers
# =====================================================================

# Novel templates: same meaning as training templates but different wording
_UN_NOVEL = [
    "the word {derived} means not {base}",
    "{base} and {derived} are opposites",
    "to be {derived} is to lack being {base}",
]

_DIS_NOVEL = [
    "the word {derived} means not {base}",
    "{base} and {derived} are opposites",
    "to {derived} is the reverse of to {base}",
]

_ING_NOVEL = [
    "the gerund of {base} is {derived}",
    "right now I am {derived} which means I {base}",
    "while {derived} one must {base} with care",
]

_ED_NOVEL = [
    "in the past we {derived} but now we {base}",
    "having {derived} once they chose to {base} again",
    "he {derived} yesterday and will {base} tomorrow",
]

_NEG_NOVEL = [
    "the word {derived} means not {base}",
    "{base} and {derived} are antonyms",
    "to be {derived} is to fail to be {base}",
]

_ER_NOVEL = [
    "more {base} means {derived}",
    "{derived} describes something more {base} than another",
    "between the two the {derived} one was more {base}",
]

_EST_NOVEL = [
    "the most {base} is the {derived}",
    "{derived} describes the most {base} of all",
    "among them all the {derived} was the most {base}",
]

_PLURAL_NOVEL = [
    "more than one {base} gives us {derived}",
    "a group of {base} is called {derived}",
    "several {derived} were seen but only one {base} remained",
]

_AGENT_NOVEL = [
    "one who {base}s is a {derived}",
    "the job of a {derived} is to {base}",
    "a professional {derived} knows how to {base} well",
]

_LY_NOVEL = [
    "in a {base} way means {derived}",
    "to do something {derived} is to be {base} about it",
    "acting {derived} shows a {base} character",
]

_RE_NOVEL = [
    "doing it over means to {derived} what you {base}",
    "a second attempt to {base} is to {derived}",
    "once more they chose to {derived} rather than just {base}",
]

_NESS_NOVEL = [
    "the quality of being {base} is called {derived}",
    "{derived} describes the state of being {base}",
    "such {derived} reveals how {base} things have become",
]

_OVER_NOVEL = [
    "doing too much of {base} leads to {derived}",
    "to {derived} is to {base} excessively",
    "excessive {base} is known as {derived}",
]

_UNDER_NOVEL = [
    "not enough {base} leads to {derived}",
    "to {derived} is to {base} insufficiently",
    "insufficient {base} means to {derived}",
]

_MIS_NOVEL = [
    "to {derived} is to {base} incorrectly",
    "a wrong {base} is a {derived}",
    "doing {base} badly is called {derived}",
]

_PRE_NOVEL = [
    "doing {base} early means to {derived}",
    "to {derived} is to {base} in advance of need",
    "preparation involves to {derived} before you {base} officially",
]

ALL_CATEGORIES = [
    (_UN_TRAIN_PAIRS, _UN_VAL_PAIRS, _UN_TEMPLATES, _UN_NOVEL),
    (_DIS_TRAIN_PAIRS, _DIS_VAL_PAIRS, _DIS_TEMPLATES, _DIS_NOVEL),
    (_ING_TRAIN_PAIRS, _ING_VAL_PAIRS, _ING_TEMPLATES, _ING_NOVEL),
    (_ED_TRAIN_PAIRS, _ED_VAL_PAIRS, _ED_TEMPLATES, _ED_NOVEL),
    (_NEG_TRAIN_PAIRS, _NEG_VAL_PAIRS, _NEG_TEMPLATES, _NEG_NOVEL),
    (_ER_TRAIN_PAIRS, _ER_VAL_PAIRS, _ER_TEMPLATES, _ER_NOVEL),
    (_EST_TRAIN_PAIRS, _EST_VAL_PAIRS, _EST_TEMPLATES, _EST_NOVEL),
    (_PLURAL_TRAIN_PAIRS, _PLURAL_VAL_PAIRS, _PLURAL_TEMPLATES, _PLURAL_NOVEL),
    (_AGENT_TRAIN_PAIRS, _AGENT_VAL_PAIRS, _AGENT_TEMPLATES, _AGENT_NOVEL),
    (_LY_TRAIN_PAIRS, _LY_VAL_PAIRS, _LY_TEMPLATES, _LY_NOVEL),
    (_RE_TRAIN_PAIRS, _RE_VAL_PAIRS, _RE_TEMPLATES, _RE_NOVEL),
    (_NESS_TRAIN_PAIRS, _NESS_VAL_PAIRS, _NESS_TEMPLATES, _NESS_NOVEL),
    (_OVER_TRAIN_PAIRS, _OVER_VAL_PAIRS, _OVER_TEMPLATES, _OVER_NOVEL),
    (_UNDER_TRAIN_PAIRS, _UNDER_VAL_PAIRS, _UNDER_TEMPLATES, _UNDER_NOVEL),
    (_MIS_TRAIN_PAIRS, _MIS_VAL_PAIRS, _MIS_TEMPLATES, _MIS_NOVEL),
    (_PRE_TRAIN_PAIRS, _PRE_VAL_PAIRS, _PRE_TEMPLATES, _PRE_NOVEL),
]


def get_train_sentences():
    """All morphological training sentences."""
    sents = []
    for train_pairs, _, templates, _ in ALL_CATEGORIES:
        for template in templates:
            sents.extend(_expand(template, train_pairs))
    return sents


def get_train_with_pairs():
    """Training sentences with word pairs."""
    items = []
    for train_pairs, _, templates, _ in ALL_CATEGORIES:
        for template in templates:
            items.extend(_expand_with_pairs(template, train_pairs))
    sents, pairs = zip(*items)
    return list(sents), list(pairs)


def get_val_sentences():
    """Validation: held-out words, same templates as training."""
    sents = []
    for _, val_pairs, templates, _ in ALL_CATEGORIES:
        for template in templates:
            sents.extend(_expand(template, val_pairs))
    return sents


def get_val_with_pairs():
    """Validation sentences with word pairs."""
    items = []
    for _, val_pairs, templates, _ in ALL_CATEGORIES:
        for template in templates:
            items.extend(_expand_with_pairs(template, val_pairs))
    sents, pairs = zip(*items)
    return list(sents), list(pairs)


def get_novel_template_sentences():
    """Validation: held-out words AND novel templates not seen in training."""
    sents = []
    for _, val_pairs, _, novel_templates in ALL_CATEGORIES:
        for template in novel_templates:
            sents.extend(_expand(template, val_pairs))
    return sents


def get_novel_with_pairs():
    """Novel template sentences with word pairs."""
    items = []
    for _, val_pairs, _, novel_templates in ALL_CATEGORIES:
        for template in novel_templates:
            items.extend(_expand_with_pairs(template, val_pairs))
    sents, pairs = zip(*items)
    return list(sents), list(pairs)
