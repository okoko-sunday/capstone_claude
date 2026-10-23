"""Classes removed from the negative ("neither") sources, and why.

Rule: an image may only be labelled "neither" if a careful human would say
it contains neither a cat nor a dog. Anything that would teach the model
"this dog is not a dog" is removed. Anything merely *animal* stays: a
rooster, a goat or a llama is exactly the kind of "neither" the instructor's
photos will contain, so the model must learn that "animal" does not mean
"pet".

Names are matched case-insensitively against the dataset's own class list,
and an unknown name raises an error, so a typo cannot silently let a dog
breed through.
"""

CALTECH101_EXCLUDE: dict[str, str] = {
    "dalmatian": "A dog breed. Labelling it 'neither' teaches the model dogs are not dogs.",
    "snoopy": "Cartoon dog. Ambiguous ground truth; the brief's test photos are real photographs.",
    "garfield": "Cartoon cat. Same reasoning as snoopy.",
    "wild_cat": "Many images are tabby-like small wild cats that are visually indistinguishable from domestic cats.",
    "background_google": "Random Google results used as clutter; unlabelled, and a spot check finds animals including pets.",
}

# Kept on purpose (documented so the decision is visible, not accidental):
CALTECH101_KEEP_NOTES: dict[str, str] = {
    "leopards": "Big wild cat, not a house cat. Kept as 'neither': the question we answer is 'is there a pet cat?'.",
    "cougar_body": "Large wild feline. Kept as 'neither' for the same reason as leopards.",
    "cougar_face": "Close-up wild feline faces. Kept, but a known source of cat/neither confusion — watch this in the matrix.",
    "faces": "People. Kept: people are a very likely 'neither' in real photographs.",
    "faces_easy": "People (tightly cropped). Kept.",
    "rooster": "Farm animal common in Nigerian surroundings. Kept as a valuable hard negative.",
    "llama": "Four-legged furry animal. Kept as a hard negative.",
}

# Imagenette (10-class ImageNet subset). One class is a dog breed.
IMAGENETTE_EXCLUDE: dict[str, str] = {
    "n02102040": "English springer spaniel — a dog breed.",
}
