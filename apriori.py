import sys
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser


def subsets(arr):
    
    all_subsets = []
    for r in range(1, len(arr) + 1):
        all_subsets.extend(combinations(arr, r))
    return all_subsets


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):

    _itemSet = set()

    # Count the support for each item in the itemSet
    localSet = defaultdict(int)
    for transaction in transactionList:
        for item in itemSet:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1

    # Filter items based on minimum support
    totalTransactions = len(transactionList)
    for item, count in localSet.items():
        support = float(count) / totalTransactions

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet


def joinSet(itemSet, length):
    
    return set(
        i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length
    )



def getItemSetTransactionList(data_iterator):
    transactionList = []
    itemSet = set()

    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        itemSet.update(frozenset([item]) for item in transaction)  # Generate 1-itemSets

    return itemSet, transactionList



def runApriori(data_iter, minSupport, minConfidence):
    
    itemSet, transactionList = getItemSetTransactionList(data_iter)

    freqSet = defaultdict(int)
    largeSet = dict()

    assocRules = dict()

    oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)

    currentLSet = oneCSet
    k = 2
    while currentLSet != set([]):
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(
            currentLSet, transactionList, minSupport, freqSet
        )
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item, freqSet, transactionList):
    
        if item not in freqSet:
            return 0.0

        return freqSet[item] / len(transactionList)


    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    toRetRules = []
    for key, value in list(largeSet.items())[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = getSupport(item) / getSupport(element)
                    if confidence >= minConfidence:
                        toRetRules.append(((tuple(element), tuple(remain)), confidence))
    return toRetItems, toRetRules


def printResults(items, rules):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    for item, support in sorted(items, key=lambda x: x[1]):
        print("item: %s , %.3f" % (str(item), support))
    print("\n------------------------ RULES:")
    for rule, confidence in sorted(rules, key=lambda x: x[1]):
        pre, post = rule
        print("Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence))


def to_str_results(items, rules):
    """returns the generated itemsets and confidence rules along with interpretations"""
    i, r, original_rules = [], [], []

    # Interpretation for items
    for item, support in sorted(items, key=lambda x: x[1]):
        interpretation = "This itemset indicates the presence of: " + ', '.join(item)
        i.append((interpretation, support))

    # Interpretation for rules
    for rule, confidence in sorted(rules, key=lambda x: x[1]):
        pre, post = rule
        interpretation = f"If {pre} occurs, then {post} is likely to occur"
        r.append((interpretation, confidence))
        original_rules.append((pre, post, confidence))

    return i, r, original_rules

def temporalAnalysis(rules, confidence_threshold, temporal_periods):
    """
    Perform iterative temporal analysis on rules and print interpretations.
    """
    for temporal_period in temporal_periods:
        print(f"\nTemporal Analysis for {temporal_period}:")

        # Filter rules based on confidence threshold and temporal period
        filtered_rules = [
            rule for rule in rules
            if float(rule.split(" , ")[1]) >= confidence_threshold and temporal_period in rule
        ]

        # Interpretation of filtered rules
        for rule_str in filtered_rules:
            # Extract information from the rule string
            rule_parts = rule_str.split(" ==> ")
            antecedent_str = rule_parts[0].replace("Rule: (", "").replace(")", "")
            consequent_str = rule_parts[1].split(" , ")[0].replace("(", "").replace(")", "")
            confidence = float(rule_parts[1].split(" , ")[1])

            # Convert antecedents and consequents to lists
            antecedents = [item.strip("'") for item in antecedent_str.split(", ")]
            consequents = [item.strip("'") for item in consequent_str.split(", ")]

            # Interpretation
            antecedent_description = ', '.join(antecedents)
            consequent_description = ', '.join(consequents)
            print(f"If {antecedent_description} occurs, then {consequent_description} is likely to occur with confidence {confidence:.3f}")

def dataFromFile(file):
    """Function which reads from the file and yields a generator"""
    for line in file:
        line = line.strip().rstrip(",")  # Remove trailing comma
        record = frozenset(line.split(","))
        yield record


if __name__ == "__main__":

    optparser = OptionParser()
    optparser.add_option(
        "-f", "--inputFile", dest="input", help="filename containing csv", default=None
    )
    optparser.add_option(
        "-s",
        "--minSupport",
        dest="minS",
        help="minimum support value",
        default=0.15,
        type="float",
    )
    optparser.add_option(
        "-c",
        "--minConfidence",
        dest="minC",
        help="minimum confidence value",
        default=0.6,
        type="float",
    )

    (options, args) = optparser.parse_args()

    inFile = None
    if options.input is None:
        inFile = sys.stdin
    elif options.input is not None:
        inFile = dataFromFile(options.input)
    else:
        print("No dataset filename specified, system with exit\n")
        sys.exit("System will exit")

    minSupport = options.minS
    minConfidence = options.minC

    items, rules = runApriori(inFile, minSupport, minConfidence)

    printResults(items, rules)
