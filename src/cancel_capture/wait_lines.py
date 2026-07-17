from __future__ import annotations

from random import Random

# These are one-line asides shown to the user while a long-running request is in flight. Keep them
# short (single line), a little stupid and a little thoughtful, and tuned per stage so the tone
# tracks whatever the model is actually doing. Edit freely; the module is data-only.
NARRATIVE_STAGE_LINES: dict[str, tuple[str, ...]] = {
    "news": (
        "Scrolling the news so you don't have to.",
        "Trying to remember what year it is.",
        "Cross-referencing three suspicious tabs.",
        "Fact-checking my own vibes.",
        "Doomscrolling, but professionally.",
        "Chasing hyperlinks like a squirrel.",
        "Sorting opinion from thing-that-happened.",
        "Assembling a small paper airplane of facts.",
        "Waiting for a headline to stop shouting.",
        "Following the citations like breadcrumbs.",
        "Yes, that also happened this week. Sorry.",
        "Reading the internet one careful sip at a time.",
    ),
    "drafting": (
        "Warming up the prose engine.",
        "Practicing the first sentence in my head.",
        "Deciding which pronoun does the least damage.",
        "Trying not to sound like a manifesto.",
        "Rehearsing a dumb pun in case it lands.",
        "Balancing dread and warmth on a small spoon.",
        "Choosing between two adjectives, forever.",
        "Aging myself five years for research purposes.",
        "Letting the anchor sign do the heavy lifting.",
        "Filing off any resemblance to real television.",
        "Squinting at the shape of five years from here.",
        "Doing the whole story in one very long breath.",
        "Making the future feel plausible and cheap.",
        "Sorting the future into small manageable disasters.",
        "Feeling one narrator emotion at a time.",
        "The metaphor is still loading. Almost there.",
    ),
    "saving": (
        "Wrapping the story in tinfoil.",
        "Sliding it into the archive drawer.",
        "Making a copy for future us.",
        "Filing this under 'ask again in five years'.",
    ),
}


def wait_lines_for(stage: str) -> tuple[str, ...]:
    if stage not in NARRATIVE_STAGE_LINES:
        raise KeyError(f"Unknown wait-line stage: {stage!r}")
    return NARRATIVE_STAGE_LINES[stage]


def random_wait_line(stage: str, *, rng: Random) -> str:
    return rng.choice(wait_lines_for(stage))
