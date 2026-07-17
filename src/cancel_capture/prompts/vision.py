VISION_SYSTEM_PROMPT = (
    "Inspect documentary photographs for prohibition signs. A qualifying sign is "
    "round, has a red border, and visibly prohibits something with a red diagonal "
    "slash or crossing. Return every distinct qualifying sign, including partial "
    "but recognizable signs. Do not return ordinary traffic signs, red circles "
    "without a prohibition mark, logos, or decorative circles. Bounding boxes use "
    "normalized image coordinates from 0 to 1. Describe only visible facts; do not "
    "invent location, intent, or text. Order signs top-to-bottom then left-to-right."
)

VISION_USER_PROMPT = (
    "Describe the complete scene factually and locate every prohibition sign. "
    "Transcribe visible sign text exactly when legible."
)
