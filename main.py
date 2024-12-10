
"""The same rules I've used for the rest of my code.
1. Variables are in PascalCase with a leading "V" to denote their variable nature.
    Dictionaries, Arrays, Lists, Sets, Tuples, and Maps all replace this V with the first letter of their type name.
2. Functions and methods are both in snake_case with a leading "f_" or "m_"
3. Classes get CAPS_SNAKE_CASE all to themselves, and variables unique to the class itself use VC instead of V to disambiguate them from the variables within instances.
4. Defaulting to PascalCase without anything leading for everything else.
5. Modules always get renamed to two-letter lowercase names to identify them.
I'm sure anything more will be too much for me keep track of effectively.
==============================
Lily Sage.
main.py
Final project: Dice Formula parser, statistical analyst, and mathematically random dice roller."""

# Import a bunch of neccessary libraries. I'm making sure to always use two-letter lowercase names for modules, so I'm adding that as another rule. Hopefully those won't get out of control lmao.
import random as ra
import re
import numpy as np
import secrets as sc
import tkinter as tk
from breezypythongui import EasyFrame as ef
import math as mt

"""Dice Collection class."""
# Now that I've actually started working with them a bit I've decied to cheat my rules a little by using VC instead of V at the start of some variables when defining a class to disambiguate between instances and the class as a whole.
# The rules are basically unchanged, this just makes the distinction readable.
class DICE_COLLECTION:
    def __init__(VSelf, VCNumDice, VCDieMax, VCMod = 0):
        VSelf.VNumDice = VCNumDice
        # Sorry Max...
        VSelf.VDieMax = VCDieMax
        VSelf.VMod = VCMod

        """Calculation Corner."""
        # This is for calculating all of the things I need for statistical analysis.

        # Calculate max, min, and average. Since these are simple calculations I'm not bothering to give them their own methods.
        VSelf.VMax = (VSelf.VDieMax * VSelf.VNumDice) + VSelf.VMod
        VSelf.VMin = VSelf.VNumDice + VSelf.VMod
        # Technically this is the mean, but I don't care. This is supposed to be a silly dice program and I'm pretty sure I talked myself into implementing an algorithm for quantum mechanics or at least some kind of fancy statistics to optimize it.
        # It's not gonna matter what kind of average this is, and this one is easy to calculate.                                                                                                            Okay so quantum mechanics is just gambling and
        VSelf.VAvg = (VSelf.VMax + VSelf.VMin)/2                                                                                                                                                          # gambling is just statistics, but you get what I mean.

        # More complex stuff, some of it handled in methods.
        VSelf.VVar = VSelf.m_calc_var()
        VSelf.VStd = mt.sqrt(VSelf.VVar)
    
    """Method for rolling the Dice Collection."""
    def m_roll(VSelf):
        # Sticking with my rules for a joke (V_). I'm using the secrets module here because I want everything to be truly random. I was originally going to do this whole thing where I made my own perfect randomness, but this is much simpler.
        # Basically this just makes a perfectly random roll for every dice in the collection, then totals it and adds the modifier.
        return (sum(sc.randbelow(VSelf.VDieMax) + 1 for V_ in range(VSelf.VNumDice)) + VSelf.VMod)

    """Calculate Variance."""
    # Variance is basically a measure of how "wide" the statistical range is, for lack of a better term. It's how much the random outcomes tend to spread out from the mean.
    def m_calc_var(VSelf):
        return VSelf.VNumDice * (((VSelf.VDieMax - 1) ** 2) / 12)

    """Calculate Probability."""
    # This uses equations that assume a bell curve and statistical properties derived from the basic information of the Dice Collection to approximate the value for a bell curve at the given VRoll value.
    # This is orders of magnitude more efficient than brute-force convolution.
    # Even using a Fast Fourier Transform, I can only get convolution down to O((n - 1) * m log(m)), but this is borderline O(m). Tricks like these can't always be used, but they should let me cut out a lot of that convolution.
    # This is also less memory intensive, since convolution will require me to store basically the same statistical distribution over and over.
    def m_calc_prob(VSelf, VRoll):
        # Using a trick I saw in an actual meme for readability.
        VUnderMin = VRoll < VSelf.VMin
        VOverMax = VRoll > VSelf.VMax
        # Don't do math if there's no chance in the first place.
        if VUnderMin or VOverMax:
            return 0
        # Extremely weird set of equations I had to look up three times, made slightly more readable through the power of VBruh.
        VBruh = mt.sqrt(2 * mt.pi)
        VCoeff = 1 / (VSelf.VStd * VBruh)
        VExp = (-(VRoll - VSelf.VAvg) ** 2)/(2 * VSelf.VVar)
        # This looks slightly insane but basically this uses the square root of Tau (2pi) and e to calculate a perfectly smooth bell curve between too extremes with variance matching the calculated variance of the Dice Collection and a total area of 1, and then calculate the float height at the given VRoll value.
        return VCoeff * mt.exp(VExp)
    
    """Probability Mass Function Calculator."""
    # The probability mass function is a dictionary containing the chance to roll any given outcome of the Dice Collection.
    def m_calc_pmf(VSelf):
        DPMF = {}
        # A singular die has a much simpler statistical distribution: Equal chances for every number on the die. Since VMod isn't counted for this, VDieMax comes in handy again.
        VSing = VSelf.VNumDice == 1
        if VSing:
            for VRoll in range(VSelf.VMin, VSelf.VMax + 1):
                DPMF[VRoll] = 1 / VSelf.VDieMax
        # Non-error non-singular behavior:
        elif VSelf.VNumDice > 1:
            VTotalP = 0
            for VRoll in range(VSelf.VMin, VSelf.VMax + 1):
                VProb = VSelf.m_calc_prob(VRoll)
                DPMF[VRoll] = VProb
                VTotalP += VProb
            # Normalization. Basically making sure that the entire probability distribution sums to 1.
            for VRoll in DPMF:
                DPMF[VRoll] /= VTotalP
        else:
            raise ValueError('Why did you even try to roll a negative number of dice?')
        return DPMF

    """Class method set up so I can directly parse tokens signifying a Dice Collection."""
    @classmethod
    def m_instance_from_token(VClass, VDiceToken):
        # This is so confusing but apparently regex match lists are their own object so I should still use V? I might have to add a rule for that. This just uses a regex thing to take a token and cut it into the integers I want.
        VMatch = re.fullmatch(r'(\d+)d(\d+)([+-]\d+)?', VDiceToken)
        # I don't need to do much input validation here because my tokenization system will do that ahead of time.
        if VMatch:
            VCNumDice = int(VMatch.group(1))
            VCDieMax = int(VMatch.group(2))
            # Fancy if statemet for assigning the modifier in one line.
            VCMod = int(VMatch.group(3) if VMatch.group(3) else 0)
            return VClass(VCNumDice, VCDieMax, VCMod)
        else:
            # I have tried to use this trick in like 3 other projects and given up but this time it works for some reason, I'm not gonna look a gift horse in the mouth, the fact that my regex works in the first place is already miracle enough.
            raise ValueError(f"Invalid Dice Collection format: {VDiceToken}")

VDSix = DICE_COLLECTION.m_instance_from_token('1d6')
VTwoDSix = DICE_COLLECTION.m_instance_from_token('2d6')

"""Dice Pool class."""
# Quite possibly the most complex class in the entire project
class DICE_POOL:
    def __init__(VSelf, VCThreshold, VCSuccesses, VCRepetitions, LCExpression):
        VSelf.VThreshold = VCThreshold
        VSelf.VSuccesses = VCSuccesses
        VSelf.VRepetitions = VCRepetitions
        VSelf.LExpression = LCExpression
    
    """Method to roll the dice pool."""
    def m_dp_roll(VSelf):
        VHits = 0
        for V_ in range (VSelf.VRepetitions):
            VRoll = 0 #Placeholder because the roll method will not be able to function until I can evaluate the dice pool's expressions.
    
    @classmethod
    def m_instance_from_token(VCLass, VDicePoolToken):
        VPoolMatch = re.fullmatch(r'dp\((\d+),\s*(\d+),\s*(\d+),\s*(.+)\)', VDicePoolToken)
        if VPoolMatch:
            VCThreshold = int(VPoolMatch.group(1))
            VCSuccesses = int(VPoolMatch.group(2))
            VCRepetitions = int(VPoolMatch.group(3))
            VCExpressionStr = VPoolMatch.group(4)

            # Tokenize and parse the nested expression
            LCNestedTok = f_tokenize(VCExpressionStr)
            LCExpression = f_parse(LCNestedTok)

            return VClass(VCThreshold, VCSuccesses, VCRepetitions, LCExpression)
        else:
            raise ValueError(f"Invalid Dice Pool format: {VDicePoolToken}")

"""Tokenization Function"""
def f_tokenize(VStr):
    # Define a truly mentally ill regex system.
    # These are each different tokens defined by a specific pattern. This is functionally a programming language of sorts.
    # I hate doing comments like these, but the alternative is making this even longer or even less readable.
    # I've made all of the names full caps because I have no idea how many of these are going to turn into their own objects, and how I'm parsing them, yeah I have nothing else to call these things. The alternative is re-casing the types that aren't classes at the end, and that's just going to be confusing.
    LTokPatterns = [
        (r'dp\(\d+,\s*\d+,\s*\d+,\s*[^,]+\)(?:[+-]\d+)?', 'DICE_POOL'), # Dice Pools are dp(x, y, z, EXPR)
        (r'\d+d\d+(?:[+-]\d+)?', 'DICE_COLLECTION'),                    # Dice Collections are defined as SomeValue d SomeOtherValue, optionally + or - Some3rdValue.
        (r'[A-Za-z!]+\(.+?\)', 'FUNCTION'),                            # Matches functions like H(x, y, EXPR) or !(EXPR)
        (r'[+\-*/^]', 'OPERATOR'),                                      # Matches +, -, *, /, ^. These are general operators.
        (r'\d+', 'NUMBER'),                                             # Matches numbers that aren't caught by everything else.
    ]

    # These get combined into one gigantic regex statement that nobody should have to write or parse manually.
    VTokMerged = '|'.join(f'({VPattern})' for VPattern, V_ in LTokPatterns)
    
    # cut the input string into matching patterns.
    VMatches = re.finditer(VTokMerged, VStr)

    # Tokenize those patterns into an actual output list:
    LTokens = []
    for VMatch in VMatches:
        for VI, (VPattern, VTokenType) in enumerate(LTokPatterns):
            if VMatch.group(VI + 1):
                LTokens.append((VTokenType, VMatch.group(VI + 1)))
                break
    return LTokens

LTestTokens = f_tokenize('H(3, 2, dp(4, 2, 3, 1d6+2)) + !(2d6+1)')

for VTokenType, VToken in LTestTokens:
    print(f"{VTokenType}: {VToken}")

import pprint as pp
pp.pprint(LParsedTest)

print(f"Mean: {VDSix.VAvg}")
print(f"Variance: {VDSix.VVar}")
print(f"Min Roll: {VDSix.VMin}")
print(f"Max Roll: {VDSix.VMax}")
print(f"Rolling VDSix: {VDSix.m_roll()}")

print(f"Mean: {VTwoDSix.VAvg}")
print(f"Variance: {VTwoDSix.VVar}")
print(f"Min Roll: {VTwoDSix.VMin}")
print(f"Max Roll: {VTwoDSix.VMax}")
print(f"Rolling VTwoDSix: {VTwoDSix.m_roll()}")
