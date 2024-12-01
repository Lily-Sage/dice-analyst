
# The same rules I've used for the rest of my code.
# 1. Variables are in PascalCase with a leading "V" to denote their variable nature.
    # Dictionaries, Arrays, Lists, Sets, Tuples, and Maps all replace this V with the first letter of their type name.
# 2. Functions and methods are both in snake_case with a leading "f_" or "m_"
# 3. Classes get CAPS_SNAKE_CASE all to themselves.
# 4. Defaulting to PascalCase without anything leading for everything else.
# I'm sure anything more will be too much for me keep track of effectively.
# ==============================
# Lily Sage.
# main.py
# Final project: Dice Formula parser, statistical analyst, and mathematically random dice roller.

# Import a bunch of neccessary libraries.
import random as ra
import re
import numpy as np
import secrets as sc

# Dice Collection class. Now that I've actually started working with them a bit I've decied to cheat my rules a little by using VC instead of V at the start of some variables when defining a class to disambiguate between instances and the class as a whole.
# The rules are basically unchanged, this just makes the distinction readable.
class DICE_COLLECTION:
    def __init__(VSelf, VCNumDice, VCDieMax, VCMod = 0):
        VSelf.VNumDice = VCNumDice
        # Sorry Max...
        VSelf.VDieMax = VCDieMax
        VSelf.VMod = VCMod
    
    # Method for rolling the Dice Collection.
    def m_roll(VSelf):
        # Sticking with my rules for a joke. I'm using the secrets module here because I want everything to be truly random. I was originally going to do this whole thing where I made my own perfect randomness, but this is much simpler.
        VTotal = sum(sc.randbelow(VSelf.VDieMax) + 1 for V_ in range(VSelf.VNumDice))
        return VTotal

    #Class method so I can kind of treat this like a variable!
    @classmethod
    def m_instance_from_token(VClass, VDiceToken):
        # This is so confusing but apparently regex match lists are their own object so I should still use V? I might have to add a rule for that. This just uses a regex thing to take a token and cut it into the integers I want.
        VMatch = re.fullmatch(r'(\d+)d(\d+)([+-]\d+)?', VDiceToken)
        if VMatch:
            VCNumDice = int(VMatch.group(1))
            VCDieMax = int(VMatch.group(2))
            # Fancy if statemet for assigning the modifier in one line.
            VCMod = int(VMatch.group(3) if VMatch.group(3) else 0)
            return VClass(VCNumDice, VCDieMax, VCMod)
        else:
            # I have tried to use this trick in like 3 other projects and given up but this time it works for some reason, I'm not gonna look a gift horse in the mouth, the fact that my regex works in the first place is already miracle enough.
            raise ValueError(f"Invalid Dice Collection format: {dice_string}")

VDSix = DICE_COLLECTION.m_instance_from_token('1d6')
print(f"Rolling VDSix: {VDSix.m_roll()}")