
"""The same rules I've used for the rest of my code.
1. Variables are in PascalCase with a leading "V" to denote their variable nature.
    Dictionaries, Arrays, Lists, Sets, Tuples, and Maps -generally just any collection type variable- all replace this V with the first letter of their type name.
        If these stack, the variable name should include the stacked collection types in order.
            If that is impossible because the collection is being built up over time, then the variable name must contain "Tree" and start with the initial of the base dictionary type as an all-else-fails proceed with caution sign.
2. Functions and methods are both in snake_case with a leading "f_" or "m_"
3. Classes get CAPS_SNAKE_CASE all to themselves, and variables unique to the class itself use VC instead of V to disambiguate them from the variables within instances.
4. Defaulting to PascalCase without anything leading for everything else.
5. Modules always get renamed to two-letter lowercase names to identify them.
I'm sure anything more will be too much for me keep track of effectively.
==============================
Lily Sage.
main.py
Final project: Dice Formula parser, statistical analyst, and mathematically random dice roller."""

"""Import Libraries"""
# Import a bunch of neccessary libraries. I'm making sure to always use two-letter lowercase names for modules, so I'm adding that as another rule. Hopefully those won't get out of control lmao.
# Random number generation. If I have a chance I'll swtich secrets to a single random call fired at program start and use that to set seeds for random instead, since that feels like a more efficient road to functionally true randomness than using secrets every time.
import random as ra
# import secrets as sc  I didn't have a chance to do that and secrets wasn't useful.
# Regex for tokenization and token parsing.
import re
# Math and statistics. These are going to be vital for statistical analysis.
# import numpy as np    #Also no longer in use.
import math as mt
# Iteration tools. Only found this when I started planning out the statistical analysis system but it should help me do some cool iteration stuff.
import itertools as it
# GUI stuff.
import tkinter as tk
from breezypythongui import EasyFrame as ef
# Multithreading to add joke popups.
import threading as th

#Genuinely I should probably go back through and drop half of these some day. I kept finding libraries to solve a problem, then solving the problem and forgetting about them. Either I'm sitting on tools to make this program a hundred times more efficient or I'm just bloating the thing, and I have no clue as to which.

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

"""Dice Pool class."""
# Quite possibly the most complex class in the entire project
class DICE_POOL:
    def __init__(VSelf, VCThreshold, VCSuccesses, VCRepetitions, LTCExp, VCMod = 0):
        VSelf.VThreshold = VCThreshold
        VSelf.VSuccesses = VCSuccesses # I had originally meant this to be useful, but it's gonna have to be left unused.
        VSelf.VRepetitions = VCRepetitions
        VSelf.LTExp = LTCExp
        VSelf.VMod = VCMod

    # Simple method to grab the dice pools tokens for parsing. This is a complete hack, but it's better than having to rewrite regex and somehow give the tokens back or implement new regex mid parsing.
    def m_dp_tokg(VSelf):
        return VSelf.LTExp

    def m_calc_pmf(VSelf, DExprPMF):
        # Setting these up so I don't have to expend the mental energy being redundant.
        VThreshold = VSelf.VThreshold
        VRepetitions = VSelf.VRepetitions
        VMod = VSelf.VMod
        
        # This calculation is based on some binomial stuff I had to read wikipedia about like seven times, but the end result is worth it.
        VSuccessProb = sum(VProb for VRoll, VProb in DExprPMF.items() if VRoll >= VThreshold)
        VFailureProb = 1 - VSuccessProb

        # Calculate the probability for different numbers of successes.
        DPMF = {}
        TotalProb = 0
        for VSuccesses in range(VRepetitions + 1):
            # Binomial probability formula in case I feel the need to look this up again.
            # P(k successes) = C(n, k) * p^k * (1-p)^(n-k)
            VProb = (
                mt.comb(VRepetitions, VSuccesses)  # Binomial coefficient C(n, k)
                * (VSuccessProb ** VSuccesses)     # p^k
                * (VFailureProb ** (VRepetitions - VSuccesses))  # (1-p)^(n-k)
            )
            DPMF[VSuccesses + VMod] = VProb
            TotalProb += VProb

        # Normalize the probabilities to make sure they're percentages.
        for VKey in DPMF:
            DPMF[VKey] /= TotalProb

        return DPMF

    """Tokenization Method"""
    @classmethod
    def m_instance_from_token(VClass, VDicePoolToken):
        VPoolMatch = re.fullmatch(r'dp\((\d+),\s*(\d+),\s*(\d+),\s*(.+?)\)([+-]\d+)?', VDicePoolToken)
        if VPoolMatch:
            VCThreshold = int(VPoolMatch.group(1))
            VCSuccesses = int(VPoolMatch.group(2))
            VCRepetitions = int(VPoolMatch.group(3))
            VCExpStr = VPoolMatch.group(4)
            VCMod = int(VPoolMatch.group(5)) if VPoolMatch.group(5) else 0

            # Tokenize for parsing. This way we have a nice neat list of tokens we can 
            LTCExp = f_tokenize(VCExpStr) # Had to tweak this and everything connected to it.

            return VClass(VCThreshold, VCSuccesses, VCRepetitions, LTCExp, VCMod) # type: ignore
        else:
            raise ValueError(f"Invalid Dice Pool format: {VDicePoolToken}")

"""Dice Function Class."""
# This is just supposed to be so I can easily track the base values of each function and validate them.
class DICE_FUNCTION:
    def __init__(VSelf, VCFunction = 'CONTAINER', VCX = 0, VCY = 0, VCZ = 0, LTCExp = []):
        VSelf.VFunction = VCFunction
        VSelf.VX = VCX
        VSelf.VY = VCY
        VSelf.VZ = VCZ
        VSelf.LTExp = LTCExp

    def m_df_tokg(VSelf):
        return VSelf.LTExp
    
    ##########
    """Dice Function PMF Calculation Methods"""
    # Since each function does something different, I need a lot of them.
    
    """H()"""
    # H(x, y, EXPR) repeats expression EXPR x times and returns the y highest rolls.
    def m_calc_h_pmf(VSelf, LExprPMF):
        VX, VY = VSelf.VX, VSelf.VY #Finally using the stuff from itertools. This product function is equivalent to nested for loops and lets me generate some convolutions showing the interactions between iterations quickly.
        LAllCombos = it.product(LExprPMF, repeat=VX)
        DPMF = {}
        for TCombo in LAllCombos:
            LSelected = sorted(TCombo, reverse=True)[:VY] # Sort by highest output.
            VTotal = sum(LSelected)
            VProb = 1
            for VResult in TCombo: # Use the sorted combinations to bias the probabilities of each output.
                VProb *= LExprPMF[VResult]
            DPMF[VTotal] = DPMF.get(VTotal, 0) + VProb
        TotalProb = sum(DPMF.values())
        for VRoll in DPMF:
            DPMF[VRoll] /= TotalProb
        return DPMF
    
    """L()"""
    # This is almost identical, I just don't flip sorting. I'm sure there's a clever way to make these two into one function but working that out at 5:30am is not something my brain was meant for, and this way I can just copy-paste the methods instead of trying to be clever.
    def m_calc_l_pmf(VSelf, LExprPMF):
        VX, VY = VSelf.VX, VSelf.VY
        LAllCombos = it.product(LExprPMF, repeat=VX)
        DPMF = {}
        for TCombo in LAllCombos:
            LSelected = sorted(TCombo)[:VY]
            VTotal = sum(LSelected)
            VProb = 1
            for VResult in TCombo:
                VProb *= LExprPMF[VResult]
            DPMF[VTotal] = DPMF.get(VTotal, 0) + VProb
        TotalProb = sum(DPMF.values())
        for VRoll in DPMF:
            DPMF[VRoll] /= TotalProb
        return DPMF
    
    """A()"""
    # This one's pretty simple. It's the same tricks as last time, but first there's an average of all possible outputs generated, and the combinations are sorted based on their proximity to that average.
    def m_calc_a_pmf(VSelf, LExprPMF):
        VX, VY = VSelf.VX, VSelf.VY
        VAvg = sum(k * v for k, v in LExprPMF.items())
        LAllCombos = it.product(LExprPMF, repeat=VX)
        DPMF = {}
        for TCombo in LAllCombos:
            LDistances = sorted(TCombo, key=lambda r: abs(r - VAvg))[:VY]
            VTotal = sum(LDistances)
            VProb = 1
            for VResult in TCombo:
                VProb *= LExprPMF[VResult]
            DPMF[VTotal] = DPMF.get(VTotal, 0) + VProb
        VTotalProb = sum(DPMF.values())
        for VRoll in DPMF:
            DPMF[VRoll] /= VTotalProb
        return DPMF
    
    """N()"""
    # This is again almost identical. Copy paste FTW.
    def m_calc_n_pmf(VSelf, LExprPMF):
        VX, VY = VSelf.VX, VSelf.VY
        VAvg = sum(k * v for k, v in LExprPMF.items())
        LAllCombos = it.product(LExprPMF, repeat=VX)
        DPMF = {}
        for TCombo in LAllCombos:
            LDistances = sorted(TCombo, key=lambda r: abs(r - VAvg), reverse=True)[:VY]
            VTotal = sum(LDistances)
            VProb = 1
            for VResult in TCombo:
                VProb *= LExprPMF[VResult]
            DPMF[VTotal] = DPMF.get(VTotal, 0) + VProb
        VTotalProb = sum(DPMF.values())
        for VRoll in DPMF:
            DPMF[VRoll] /= VTotalProb
        return DPMF
    
    """E()"""
    # Actually different for once. Able to use an exponent to do most of the calculations because a lot of statistics things I knew and had planned out, but cannot properly recall.
    def m_calc_e_pmf(VSelf, LExprPMF):  # Either that or the fist mental faculties I'm losing include my ability to tell how hard something is to read.
        VX = VSelf.VX
        DPMF = {}
        for VRoll, VProb in LExprPMF.items():
            if VRoll % 2 == 0:
                DPMF[VRoll] = 1 - (1 - VProb) ** VX
        VTotalProb = sum(DPMF.values())
        for VRoll in DPMF:
            DPMF[VRoll] /= VTotalProb
        return DPMF
    
    """O()"""
    # Back to copying... :D
    def m_calc_o_pmf(VSelf, LExprPMF):
        X = VSelf.VX
        DPMF = {}
        for VRoll, VProb in LExprPMF.items():
            if VRoll % 2 != 0:
                DPMF[VRoll] = 1 - (1 - VProb) ** X
        VTotalProb = sum(DPMF.values())
        for VRoll in DPMF:
            DPMF[VRoll] /= VTotalProb
        return DPMF
    
    """!!()"""
    # Another new one. Each iteration builds up on the original to simulate dice explosions.
    def m_calc_explosion_pmf(VSelf, LExprPMF):
        X, Y, Z = VSelf.VX, VSelf.VY, VSelf.VZ
        DPMF = dict(LExprPMF)
        for _ in range(Z - 1):
            DNewPMF = {}
            for VRoll, VProb in DPMF.items():
                if VRoll == X:
                    for VSubRoll, VSubProb in LExprPMF.items():
                        for _ in range(Y):
                            VResult = VRoll + VSubRoll
                            DNewPMF[VResult] = DNewPMF.get(VResult, 0) + VProb * VSubProb
                else:
                    DNewPMF[Roll] = DNewPMF.get(Roll, 0) + VProb
            VTotalProb = sum(DNewPMF.values())
            for Roll in DNewPMF:
                DNewPMF[Roll] /= VTotalProb
            DPMF = DNewPMF
        return DPMF
    
    """!()"""
    # Factorials. I'm not sure what else to say. I am commenting after getting things mostly running, so I don't have much else to say. The statistics stay the same, the values just get put through the factorial function.
    def m_calc_factorial_pmf(VSelf, LExprPMF):
        DPMF = {}
        for VRoll, VProb in LExprPMF.items():
            VFactRoll = mt.factorial(VRoll)
            DPMF[VFactRoll] = DPMF.get(VFactRoll, 0) + VProb
        return DPMF

    """()"""
    # Containers. Not even anything to comment on here. You just move on with it.
    def m_container(VSelf, DPMF):
        return DPMF
    
    """Method Select"""
    # A bunch of if statements so I don't have to figure out what function to use every time.
    def m_funk_sel(VSelf, DPMF):
        # Using a dictionary to create an artifical somewhat cheaty selection statement but god I prefer this over the slog I was planning last night.
        DPMFMeth = {
            'H': VSelf.m_calc_h_pmf,
            'L': VSelf.m_calc_l_pmf,
            'A': VSelf.m_calc_a_pmf,
            'N': VSelf.m_calc_n_pmf,
            'E': VSelf.m_calc_e_pmf,
            'O': VSelf.m_calc_o_pmf,
            '!!': VSelf.m_calc_explosion_pmf,
            '!': VSelf.m_calc_factorial_pmf,
            'CONTAINER': VSelf.m_container
        }

        if VSelf.VFunction in DPMFMeth:
            return DPMFMeth[VSelf.VFunction](DPMF)
        
    # I'm writing on sleep deprivation again and I really shouldn't be but here! Instances!
    # I realized recently that I definitely have to do validation for dice pools and collections inside their instance_from_token method. I have nearly no checks otherwise. So I'm making this one work now, and then I'm putting "go back and make the other two classes secure" on the todo list.
    @classmethod
    def m_instance_from_token(VClass, VToken):
        # Just going for the function itself and putting the arguments on the backburner. Since there are only 9 functions we can be cheeky and just match to those functions, nothing else.
        LMatches = re.fullmatch(r'(H|L|A|N|E|O|!!|!|)\((.*)\)', VToken)

        # Complain to the user if they somehow snuck an invalid function past the tokenizer.
        if not LMatches: # type: ignore
            raise ValueError(f"Invalid function format: {VFunctionToken}. How????") # type: ignore
        
        # Set functions with special escapes for the container function and empty instructions.
        VCFunction = LMatches.group(1) if LMatches.group(1) else 'CONTAINER'
        VArgsStr = LMatches.group(2) if LMatches.group(2) else ''

        # A dictionary containing the min and max acceptable number of arguments for each.
        DFunkSpec = {
            'H': (1, 3), 'L': (1, 3), 'A': (1, 3), 
            'N': (1, 3), 'E': (1, 2), 'O': (1, 2), 
            '!!': (2, 4), '!': (1, 1), 'CONTAINER': (1, 1)
        }

        # Another one for which arguments for each of the functions have defaults.
        DFunkDefC = {
            'H': (0, 1), 'L': (0, 1), 'A': (0, 1), 
            'N': (0, 1), 'E': (0), 'O': (0), 
            '!!': (2), '!': (), 'CONTAINER': ()
        }

        # And a third for those defaults. 0 values are used to mark default or unused variables.
        DFunkDefaults = {
            'H': (2, 1, 0), 'L': (2, 1, 0), 'A': (2, 1, 0), 
            'N': (2, 1, 0), 'E': (2, 0, 0), 'O': (2, 0, 0), 
            '!!': (0, 0, 1), '!': (0, 0, 0), 'CONTAINER': (0, 0, 0)
        }

        # Load defaults.
        VCX, VCY, VCZ = DFunkDefaults[VCFunction]

        if VCFunction not in DFunkSpec:
            raise ValueError(f"Invalid function name: {VCFunction}. Okay maybe just use the right function?")

        VMinArgs, VMaxArgs = DFunkSpec[VCFunction]

        """# Regex (:  Tries to cut the arguments up but avoid any collisions that might happen within the EXPR block by hopping over parenthesis.
        LArgs = re.split(r',\s*(?![^()]*\))', VArgsStr.strip()) if VArgsStr else [] # This last bit is to catch an error that should be logically impossible where the user somehow submits a function with an empty set of arguments. I refuse not to try and idiot-proof that aspect. Somehow someone will figure out how to get H() past the tokenizer and this should hopefully limit the damage."""
        # This code failed to account for nested expressions, mothballed to document the error.
        # Fixed code step 1: Loop through and forcibly pair token parenthesis. This is necessary because the tokenization system drops parenthesis at the end if they stack without some kind of token separating them.
        VArgumentBalance = 0
        for VChar in VArgsStr:
            if VChar == '(':
                VArgumentBalance += 1
            elif VChar == ')':
                VArgumentBalance -= 1
        if VArgumentBalance > 0:
            for VI in range (VArgumentBalance):
                VArgsStr += ')'
        
        # Step two: loop through and only split on commas when the balance is at 0. This should prevent nested argument parsing.
        VArgumentBalance = 0
        LArgs =[]
        LCurArgChars = []
        for VChar in VArgsStr:
            if VChar == ',' and VArgumentBalance == 0:
                LArgs.append(''.join(LCurArgChars).strip())
                LCurArgChars = []
            else:
                if VChar == '(':
                    VArgumentBalance += 1
                elif VChar == ')':
                    VArgumentBalance -= 1
                LCurArgChars.append(VChar)
        # That doesn't usually catch the last argument but that's fine because we can just do this:
        if LCurArgChars:
            LArgs.append(''.join(LCurArgChars).strip())
        
        print(LArgs)
        

        # Validate argument count. You understand why I had to do this, right?
        if not (VMinArgs <= len(LArgs) <= VMaxArgs):
            raise ValueError(f"{VCFunction} expects between {VMinArgs} and {VMaxArgs} arguments, but got {len(LArgs)}.")
        
        # Set changed values from right to left. There's only one case where there are three arguments, and in that case right to left already prefers non-defaulted values. By using greater-than here I can just used the particular argument index and automatically ensure that the function expressions aren't touched.
        # Updated: No longer use fancy defaulting trick that causes a value error when using less than 4 arguments.
        if len(LArgs) == 2:
            VCX = int(LArgs[0])
        elif len(LArgs) == 3:
            VCX = int(LArgs[0])
            VCY = int(LArgs[1])
        elif len(LArgs) == 4:
            VCX = int(LArgs[0])
            VCY = int(LArgs[1])
            VCZ = int(LArgs[2])

        # Tokenize the function's expression and store it.
        LTCExp = f_tokenize(LArgs[-1])

        return VClass(VCFunction, VCX, VCY, VCZ, LTCExp)

"""Tokenization Method"""
# Tokenization function has been emailed to the parse function for convenience.
# Actually no it hasn't this thing is load-bearing. (A fact I only realized several days after moving it.)
def f_tokenize(VFormula):
    # Define a truly mentally ill regex system.
    # These are each different tokens defined by a specific pattern. This is functionally a programming language of sorts.
    # I hate doing comments like these, but the alternative is making this even longer or even less readable.
    # I've made all of the names full caps because I have no idea how many of these are going to turn into their own objects, and how I'm parsing them, yeah I have nothing else to call these things. The alternative is re-casing the types that aren't classes at the end, and that's just going to be confusing.
    LTokPatterns = [
        (r'dp\(\d+,\s*\d+,\s*\d+,\s*[^,]+\)(?:[+-]\d+)?', 'DICE_POOL'), # Dice Pools are dp(x, y, z, EXPR)
        (r'\d+d\d+(?:[+-]\d+)?', 'DICE_COLLECTION'),                    # Dice Collections are defined as SomeValue d SomeOtherValue, optionally + or - Some3rdValue.
        (r'[A-Za-z!]+\(.+?\)', 'FUNCTION'),                             # Matches functions like H(x, y, EXPR) or !(EXPR)
        (r'[+\-*/^]', 'OPERATOR'),                                      # Matches +, -, *, /, ^. These are general operators.
        (r'\d+', 'NUMBER'),                                             # Matches numbers that aren't caught by everything else.
    ]

    # These get combined into one gigantic regex statement that nobody should have to write or parse manually.
    VTokMerged = '|'.join(f'({VPattern})' for VPattern, V_ in LTokPatterns)
       
    # cut the input string into matching patterns.
    VMatches = re.finditer(VTokMerged, VFormula)

    # Tokenize those patterns into an actual output list:
    LTTokens = []
    for VMatch in VMatches:
        for VI, (VPattern, VTokenType) in enumerate(LTokPatterns):
            if VMatch.group(VI + 1):
                # Yes this makes a list made of tuples, I have something wrong with me for calling this LTokens. This is the most confusing part of the code and it slowed me down for several days until I remembered why I did it.
                # I am now renaming it LTTokens because nesting dictionaries are going to get insanely confusing otherwise. 
                LTTokens.append((VTokenType, VMatch.group(VI + 1)))
                break
    return LTTokens

##########
"""Syntax Tree Functions Stolen from f_parse"""
# These were very useful when working with the syntax tree.
"""Recusrive Syntax Tree Actor Function"""
# Recursively iterate through the current tree, acting on each full node before moving on to its children. Tweaked to hit everything now that I know the syntax tree is complete.
def f_syntax_rec_do_inv(DCurNode, f_callback):
    # I lied! It's so much more useful now to go through the syntax tree in reverse. In all honest I had to hack this together after getting 99% of the remaining code for statistical convolution up and running so I actually had a way to run the whole thing.
    for DSubNode in DCurNode.get('LArgs', []):
        # Skip unfinished nodes.
        if isinstance(DSubNode, dict):
            f_syntax_rec_do_inv(DSubNode, f_callback)
    f_callback(DCurNode)

"""Snytax Tree Navigation Function"""
# A complete hack solution to navigating the syntax tree for editing. Just set the current node to the next address over and over until you reach your destination. Taken out of f_parse so I can use it for evaluating the completed syntax tree.
def f_syntax_nav(LAdd, DTargetTree):
    DCurrentNode = DTargetTree # Tweaked to be more general and not raise errors.
    for VIn in LAdd:
        try:
            DCurrentNode = DCurrentNode['LArgs'][VIn]
            if not isinstance(DCurrentNode, dict):
                raise TypeError(f"Expected a dictionary at address {LAdd}, but got {type(DCurrentNode)}: {DCurrentNode}")
        except (IndexError, KeyError) as VE:
            raise ValueError(f"Syntax Tree Address Out of Bounds: {LAdd}: {VE}")
    # I nearly turned this into a whole complex callback statement because I somehow missed the fact that python is nice about collections.
    # I feel like a god. I was overcomplicating this so much these past few days. It's literally this simpled, I've gone through eight versions of my parse function, and it's literally this simple.
    return DCurrentNode

# Simplifies completion a little by making it so the code doesn't have to find the ROOT VComplete Boolean over and over.
def f_is_syntax_comp(DTargetTree):
    return DTargetTree['VComplete']
##########

"""Parsing Function"""
# Oh finally getting to write this hurt a little. Two weeks trying to figure out token parsing. If my keyboard could bleed, this function would be wrtitten in its blood on the walls of my room like the scrawlings of a madwoman.
# To all who look upon my code and weep, know this: I am not sorry.
def f_parse(VFormula):
    #This is such a complex formula that I'm going to first start with about 8 load-bearing sub-function definitions and then actually write the code for the function.
    # Syntax tree with root node:       Originally this didn't use the fancier, more navigable structure I'm using now. The current pattern is: {(Token type, Token): Address, 'VValue': Value, 'VComplete': Bool, 'LArgs': Args}
    DSyntaxTree = {('ROOT', VFormula): [], 'VValue': None, 'VComplete': False, 'LArgs': []}

    """Tokenization Function"""
    # Transplanted here because now that there's a general parse function, the tokenization function isn't useful anywhere else.
    # After an accident involving unclear scope, the tokenization function was shipped back. (Removed after 2 days due to load-bearing tokenization occuring within some of the classes)

    """Syntax Layer Generation Function"""
    # My eigth attempt at making this good. It finally words.
    def f_syntax_pass(LTTokens, LAdd):
        # Navigate to the target address.
        DNode = f_syntax_nav(LAdd, DSyntaxTree)

        # Ensure LTTokens is an independent copy of itself. Stacked collection variabls secretly being pointers saved me an hour ago, so now it is seeking to do the opposite.
        LTTokens = LTTokens.copy()

        # Clear the current node's LArgs to ensure a clean slate. Without this the tokens some parsed nodes temporarily store in their LArgs list cause the code to break down.
        if 'LArgs' in DNode:
            # Trying with great fear to ensure LArgs is actually always a list? What is going on. I chased an error around for a significant amount of time and have no clue which verification thing saved it.
            if not isinstance(DNode['LArgs'], list):
                raise TypeError(f"LArgs should be a list, but got {type(DNode['LArgs'])}: {DNode['LArgs']}")
            DNode['LArgs'].clear()
        else:
            DNode['LArgs'] = []

        # Iterate through the tokenized string and generate nodes based on the tokens, itself containing sub-tokens.
        for VTokenType, VToken in LTTokens:
            VTokenIndex = LTTokens.index((VTokenType, VToken))
            LNodeAdd = LAdd + [VTokenIndex]
            DNewNode = f_make_node(VTokenType, VToken, LNodeAdd)
            # Append the new node to the target address.
            DNode['LArgs'].append(DNewNode)

    """Recusrive Syntax Tree Actor Function"""
    # Recursively iterate through the current tree, acting on each full node before moving on to its children.
    def f_syntax_rec_do(DCurNode, f_callback):
        # Using a callback here because this is supposed to be a carrier for a variety of different actions that need to hit all or many incomplete nodes.
        f_callback(DCurNode)
        for DSubNode in DCurNode.get('LArgs', []):
            # Skip unfinished nodes and nodes marked as complete.
            if isinstance(DSubNode, dict) and not DSubNode['VComplete']:
                f_syntax_rec_do(DSubNode, f_callback)
    
    """Completion Verification Function"""
    # For use with f_syntax_rec_do, keeps track of syntax tree completion.
    def f_complete_check(DCurNode):
        #It just works. Make sure the node is marked incomplete if it has incomplete arguments.
        if any(isinstance(VArg, tuple) for VArg in DCurNode.get('LArgs', [])):
            DCurNode['VComplete'] = False
            return
        # Do nothing if the node is complete, if it's not complete, check all of its children and mark it complete if they're all complete.
        elif not DCurNode['VComplete']:
            # Really glad I've started picking up tricks like this and applying them more.
            if all(isinstance(VArg, dict) and VArg.get('VComplete', False) for VArg in DCurNode.get('LArgs', [])):
                DCurNode['VComplete'] = True

    """Recursive Parsing function"""
    # Finds places where the syntax tree is incomplete and calls f_syntax_pass on the associated node. Is used with f_snytax_rec_do().
    def f_syntax_rec_pass(DCurNode):
        # :D
        if all(isinstance(VArg, tuple) for VArg in DCurNode.get('LArgs', [])):
            # Making my nodes dictionaries has been a godsend so far. I'm moving around so much information.
            LTTokens = DCurNode['LArgs']
            LAdd = list(DCurNode.values())[0]
            f_syntax_pass(LTTokens, LAdd)
    
    """Verification and parsing tick function"""
    # This lets me update the syntax tree over and over again with one instruction.
    def f_rec_tick():
        if not f_is_syntax_comp(DSyntaxTree):
            f_syntax_rec_do(DSyntaxTree, f_syntax_rec_pass)
            f_syntax_rec_do(DSyntaxTree, f_complete_check)

    def f_make_node(VTokenType, VToken, LAdd =[]):
        if VTokenType == 'DICE_COLLECTION':
            # God these methods are so useful.
            VDiceCol = DICE_COLLECTION.m_instance_from_token(VToken)
            # Add a complete node with the dice collection instance.
            
            return {
                ('DICE_COLLECTION', VToken): LAdd,
                'VValue': VDiceCol,
                'VComplete': True,
                'LArgs': []
            }

        elif VTokenType == 'DICE_POOL':
            # Originally I was going to treat these and functions with extreme caution as raw tokens to go over later. Instead I'm just going to go all out and build them later, but that doesn't actually make much sense.
            # For sheer convenience I'm going to give the Dice pool a duplicate of its own arguments: I wrote the class's regex before working out how the syntax tree would work, and it would be extremely inconvenient to go back.
            VDicePool = DICE_POOL.m_instance_from_token(VToken)
            LTDPTokens = VDicePool.m_dp_tokg()

            return {
                ('DICE_POOL', VToken): LAdd,
                'VValue': VDicePool,
                'VComplete': False,
                'LArgs': LTDPTokens
            }

        elif VTokenType == 'FUNCTION':
            VDiceFunk = DICE_FUNCTION.m_instance_from_token(VToken)
            LTDFTokens = VDiceFunk.m_df_tokg()
            return {
                ('FUNCTION', VToken): LAdd,
                'VValue': VDiceFunk,
                'VComplete': False,
                'LArgs': LTDFTokens
            }

        # Operators are so simple I can just drop them in and use them as raw tokens, there's no need for special parsing.
        elif VTokenType == 'OPERATOR':
            return {
                ('OPERATOR', VToken): LAdd,
                'VValue': VToken,
                'VComplete': True,
                'LArgs': []
            }

        elif VTokenType == 'NUMBER':
            # Validate the numbers just in case.
            try:
                VNum = abs(int(VToken))
            except:
                raise ValueError(f"Unacceptable number token {VToken}!")
            
            return {
                ('NUMBER', VToken): LAdd,
                'VValue': VNum,
                'VComplete': True,
                'LArgs': []
            }
        
        # Error handling.
        elif VTokenType == 'ROOT':
            raise ValueError('Unparsable root node in token list!')
        
        else:
            # I checked and yes this is also a value error. I went through all the trouble of looking up more error types and then whoops basically all of the errors I catch are ValueErrors.
            raise ValueError('Uncaught token type while parsing at line 666!')
        

    # Tokenize the input formula. This way I can move the tokenization function inside the parse function and make it all pretty and nested. (I ended up not doing this.)
    LTTokens = f_tokenize(VFormula)
    # Initial syntax pass to get things started.
    f_syntax_pass(LTTokens, [])
    
    # While loop that only continues while the syntax tree is incomplete:
    while not f_is_syntax_comp(DSyntaxTree):
        # Attempt to expand the syntax tree and check it for completion.
        f_rec_tick()

    return DSyntaxTree

# Instead of going through the full syntax tree and evaluating it for every roll, I'll generate and store the probability mass function of the full equation and use that to randomly generate numbers instead.
# I only need two specific functions here for the PMFs since everything else is handled inside the methods of the three classes.

"""Operator Convolution Function"""
# Convolve two input PMFs based on the assigned Operator.
def f_con_op_pmf(DPMF1, VOperator, DPMF2):
    # Blank slate.
    DResultPMF = {}

    # Use some itertools stuff and for loops to simulate the operations happening on every possible combination of the input PMFs.
    for (VOutcome1, VProb1), (VOutcome2, VProb2) in it.product(DPMF1.items(), DPMF2.items()):
        if VOperator == '+':
            VResult = VOutcome1 + VOutcome2
        elif VOperator == '-':
            VResult = VOutcome1 - VOutcome2
        elif VOperator == '*':
            VResult = VOutcome1 * VOutcome2
        elif VOperator == '/':
            if VOutcome2 != 0:  # Remember kids dividing by zero is very bad and unhealthy. At this point I think I might just be truly over-commenting. Most of it is quips.
                VResult = VOutcome1 / VOutcome2
            else:
                continue
        else:
            raise ValueError(f"Unsupported operator: {VOperator}")

        # Update the PMF with the new probability.
        DResultPMF[VResult] = DResultPMF.get(VResult, 0) + VProb1 * VProb2

    # Normalize the resulting PMF just to be safe.
    VTotalProb = sum(DResultPMF.values())
    for VRoll in DResultPMF:
        DResultPMF[VRoll] /= VTotalProb

    return DResultPMF

"""Number Convolution Function"""
# This is a very serious and normal function that absolutely has a purpose.
# No seriously this makes things a lot more convenient.
def f_number_pmf(VNumber):
    return {VNumber: 1.0}

"""Operator Convolution Evaluator"""
# Takes in a list of node arguments with known Operator nodes within them, find and evaluate PMFs based on the operators present until all operators and PMFs have been worked down into a single PMF for use in the parent node.\
# I really need to remember to use parent and child or some other consistent relational terminology when talking about these nodes.
def f_con_op_eval(LNodes):
    """Iteratively evaluate PMFs using operators."""
    DCurrentPMF = LNodes[0]['VValue']  # Start with the first argument's PMF.
    for VArgIndex in range(1, len(LNodes), 2):  # Skip to operators.
        DOperator = LNodes[VArgIndex]['VValue']  # Get the operator.
        DNextPMF = LNodes[VArgIndex + 1]['VValue']  # Get the next PMF after the operator.
        DCurrentPMF = f_con_op_pmf(DCurrentPMF, DOperator['VValue'], DNextPMF)  # Run the single operator convolution function using the current PMF and the PMF following the operator.
    return DCurrentPMF

"""General Node Convolution Evaluator"""
# Makes sure a target node is handled properly for each node type.
def f_eval_node(DCurNode):
    # Get node type.
    VNodeType, V_ = list(DCurNode.keys())[0]

    # Check if the node has Operator child nodes.
    VContainsOperator = any(
        isinstance(DSubNode, dict) and 'OPERATOR' in DSubNode for DSubNode in DCurNode.get('LArgs', [])
    )

    if VContainsOperator:
        # Evaluate using the operator convolution functions.
        DCurNode['VValue'] = f_con_op_eval(DCurNode['LArgs'])
    else:
        # Evaluate based on the node types.
        if VNodeType == 'DICE_COLLECTION':
            DCurNode['VValue'] = DCurNode['VValue'].m_calc_pmf()
        elif VNodeType == 'DICE_POOL':
            DCurNode['VValue'] = DCurNode['VValue'].m_calc_pmf(DCurNode['LArgs'][0]['VValue'])
        elif VNodeType == 'FUNCTION':
            DCurNode['VValue'] = DCurNode['VValue'].m_funk_sel(DCurNode['LArgs'][0]['VValue'])
        elif VNodeType == 'OPERATOR': # Don't do anything to operators.
            return
        elif VNodeType == 'NUMBER':
            DCurNode['VValue'] = f_number_pmf(DCurNode['VValue'])
        elif VNodeType == 'ROOT':
            DCurNode['VValue'] = DCurNode['LArgs'][0]['VValue']
        else:
            raise ValueError('Uncaught node of unknown type when evaluating syntax tree for convolution.')
    
def f_syn_tree_convolve(DSyntaxTree):
    # Iterate through the lowest levels of the syntax tree first and evaluate each one, overwriting VValue with the pmf for use in later iterations.
    f_syntax_rec_do_inv(DSyntaxTree, f_eval_node)
    return DSyntaxTree['VValue']  # Return the PMF of the root node.

def f_roll(DPMF):
    LOutcomes = list(DPMF.keys())
    LChances = list(DPMF.values())
    VResult = ra.choices(LOutcomes, weights=LChances, k=1)[0]
    return VResult

# Gotta go through this part much less precisely. It's probably gonna suck a bit, might be a bit inconsistent, but it'll work. Limited commenting from here on out.
# God this is so ugly.
# I've tried 5 different things and it's only making things worse.
# Okay nope it's starting to look a bit better.
class CORE_WINDOW(ef):
    def __init__(VSelf):
        ef.__init__(VSelf, title='DiceAnalyst', width=400, height=150, resizable=False)

        # Formula input box
        VSelf.VFormulaBox = tk.Text(VSelf, wrap='word', height=4, width=50, bg='#d3d3d3')
        VSelf.VFormulaBox.pack(pady=10)

        # Buttons
        VButtonFrame = tk.Frame(VSelf)
        VButtonFrame.pack(pady=5)

        VSelf.VEvaluateButton = tk.Button(VButtonFrame, text='Evaluate', command=VSelf.m_evaluate)
        VSelf.VEvaluateButton.pack(side='left', padx=5)

        VSelf.VHelpButton = tk.Button(VButtonFrame, text='?', command=VSelf.m_show_help)
        VSelf.VHelpButton.pack(side='left', padx=5)

        #I needed to have images sooo.
        VSelf.m_start_popup()

    def m_evaluate(VSelf):
        # Get the user input.
        VFormula = VSelf.VFormulaBox.get('1.0', tk.END).strip()

        if not VFormula:
            VSelf.messageBox('Error', 'Please enter a formula to evaluate.')
            return

        try:
            # Parse and evaluate.
            DTreeSyntax = f_parse(VFormula)
            DPMF = f_syn_tree_convolve(DTreeSyntax)

            # Display the PMF.
            VSelf.m_display_pmf_window(DPMF)
        except Exception as VError:
            # Display errors.
            VSelf.messageBox('Error', f'An error occurred:\n{VError}')

    def m_display_pmf_window(VSelf, DPMF):
        VPMFWindow = tk.Toplevel(VSelf)
        VPMFWindow.title('DiceAnalyst: Stats')
        VPMFWindow.geometry('400x800')
        VPMFWindow.resizable(False, False)

        VCanvas = tk.Canvas(VPMFWindow, width=150, height=500)
        VScrollbar = tk.Scrollbar(VPMFWindow, orient=tk.VERTICAL, command=VCanvas.yview)
        VScrollableFrame = tk.Frame(VCanvas)

        VScrollableFrame.bind(
            '<Configure>',
            lambda e: VCanvas.configure(scrollregion=VCanvas.bbox('all'))
        )

        VCanvas.create_window((0, 0), window=VScrollableFrame, anchor='nw')
        VCanvas.configure(yscrollcommand=VScrollbar.set)

        VCanvas.pack(side='left', fill=tk.BOTH, expand=True)
        VScrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for VOutcome, VProbability in sorted(DPMF.items()):
            VRow = tk.Frame(VScrollableFrame)
            VRow.pack(fill='x', pady=2)

            VLabel = tk.Label(VRow, text=f'{VOutcome}', width=5, anchor='w')
            VLabel.pack(side='left', padx=5)

            VBar = tk.Frame(VRow, bg='#d3d3d3', width=int(VProbability * 1000), height=10)
            VBar.pack(side='left', padx=5, fill='x', expand=False)

            VPercent = tk.Label(VRow, text=f'{VProbability:.2%}', anchor='w')
            VPercent.pack(side='left', padx=5)

        # Label to display the roll result
        VRollResultLabel = tk.Label(VScrollableFrame, text='Roll Result: ', font=('Arial', 12))
        VRollResultLabel.pack(side='top', pady=5)

        # Roll button
        VRollButton = tk.Button(
            VScrollableFrame, 
            text='Roll', 
            command=lambda: VSelf.m_roll(DPMF, VRollResultLabel)
        )
        VRollButton.pack(side='bottom', pady=10)

    def m_roll(VSelf, DPMF, VRollResultLabel):
        try:
            outcome = f_roll(DPMF)
            VRollResultLabel.config(text=f'Roll Result: {outcome}')
        except Exception as e:
            VSelf.messageBox('Error', f'An error occurred during the roll:\n{e}')

    def m_show_help(VSelf):
        # Create a new top-level window for the help content
        VHelpWindow = tk.Toplevel(VSelf)
        VHelpWindow.title('DiceAnalyst: Help')
        VHelpWindow.geometry('700x400')
        VHelpWindow.resizable(True, True)

        # Create a scrollable text area
        VTextArea = tk.Text(VHelpWindow, wrap='word', font=('Arial', 12), bg='#f4f4f4')
        VTextArea.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        # Add a vertical scrollbar
        VScrollbar = tk.Scrollbar(VHelpWindow, command=VTextArea.yview)
        VScrollbar.pack(side='right', fill='y')
        VTextArea.configure(yscrollcommand=VScrollbar.set)

        # Help text content
        helpText = (
            'DiceAnalyst allows you to enter formulas to evaluate probability distributions '
            'and roll simulated dice. Use the following syntax for your formulas:\n\n'
            '- Dice Pools: dp(threshold, goal(unused), repetitions, expression)\n'
            '- Dice Collections: XdY[+/-Z]\n'
            '- Functions:\n'
            '-- Function H(x, y, EXPR) repeats expression EXPR x times and returns the y highest rolls.\n'
            '--- Function L(x, y, EXPR) does the same, but with the lowest y results instead.\n'
            '-- Function A(x, y, EXPR) repeats EXPR x times and returns y results that are closest to the average.\n'
            '--- Function N(x, y, EXPR) does the same for the results furthest from the average.\n'
            '-- Function E(x, EXPR) repeats expression EXPR x times. It returns the first output that is even if there is an even output, otherwise, it returns the first odd output.\n'
            '--- Function O(x, EXPR) does the same for odd values.\n'
            '-- Function !!(x, y, z, EXPR) explodes EXPR expression on x output into y bonus iterations with recursion up to z.\n'
            '-- Function !(EXPR) calculates the factorial of whatever EXPR returns.\n'
            '-- Function (EXPR) is basic quality of life and flow control.\n'
            'Example: dp(4, 3, 6, H(2d6))'
        )

        # Insert the help text into the text area
        VTextArea.insert('1.0', helpText)
        VTextArea.config(state='disabled')  # Make the text area read-only

    # Easter egg type thing but it's joke popup ads with a couple of cat memes.
    # I got everything else out of the way so instead of thinking about how trash my GUI probably is, I'm adding cat memes.
    def m_start_popup(VSelf):
        def m_popup_task():
            while True:
                # Chill for anywhere between 3 minutes and several hours.
                VInterval = ra.randint(18, 108)
                th.Event().wait(VInterval)

                LImagePaths = ['bigstomp.png', 'pspspsps.png', 'thereisno.png']
                VSelectedImage = ra.choice(LImagePaths)

                # Show the popup.
                VSelf.after(0, lambda: VSelf.m_show_popup(VSelectedImage))

        # Run on a separate thread while waiting.
        th.Thread(target=m_popup_task, daemon=True).start()

    def m_show_popup(VSelf, VImagePath):
        VPopup = tk.Toplevel(VSelf)
        VPopup.title('A WORD FROM OUR SPONSOR')
        VPopup.geometry('600x600') # If I could I would find a way to resize these, but for some reason my version of pip is going crazy and I can't install a library that would work. And on top of that I have 3 hours to submit.
        VPopup.resizable(False, False) # If I ever come back and fix things, this will be one of them.

        try:
            VTkImage = tk.PhotoImage(file=VImagePath)
        except tk.TclError: # Error handling.
            tk.messagebox.showerror('Error', f'Failed to load image: {VImagePath}')
            return

        VLabel = tk.Label(VPopup, image=VTkImage)
        VLabel.image = VTkImage # prevent garbage collection just in case, since this could be sitting around for hours without doing anything.
        VLabel.pack(pady=10)

        VButton = tk.Button(VPopup, text='Close', command=VPopup.destroy)
        VButton.pack(pady=5)

# Main function to launch the GUI
def main():
    CORE_WINDOW().mainloop()

if __name__ == '__main__':
    main()


# A retrospective of sorts.
"""
This is the first time I have ever made a program this long or complete, and it's really taught me some things. I'm used to stuff like Unreal Engine where the logic is all there but there's a lot less typing or investment.
And on top of that I have taken a pretty significant break from actually doing game design and programming like that. I usually do creative writing and art now.
So this has been a really good exercise in shaking the rust off and learning how to program without training wheels.
I feels I've done well enough to show I know what I'm doing, at least, but every hour, maybe even every minute I spent on this taught me something. Looking back my code probably looks like eight people stiched it together because of how many times I learned to tweak things or tried to pick up a new trick to get it done faster.
This was a trial by fire, and I barely made it through, but that's alright. I'm glad I did it.
Last time I had this class I bit off more than I could chew too. I tried making an entire functioning website in around 6 days.
This was a similar challenge, but I pulled it off in the end, and unlike last time, I made sure I actually learned something while doing it.
"""