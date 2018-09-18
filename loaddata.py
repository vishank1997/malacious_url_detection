import os
import random


def load_training_data():
    """
    Load the phishing domains and benign domains from disk into python lists

    NOTE: I'm using a smaller set of samples than from the CLI tool so the feature extraction is quicker.

    @return training_data: dictionary where keys are domain names and values
                are labels (0 = benign, 1 = phishing).
    """
    training_data = {}

    benign_path = "benign/"
    for root, dirs, files in os.walk(benign_path):
        files = [f for f in files if not f[0] == "."]
        for f in files:
            with open(os.path.join(root, f)) as infile:
                for item in infile.readlines():
                    # Safeguard to prevent adding duplicate data to training set.
                    if item not in training_data:
                        training_data[item.strip('\n')] = 0

    phishing_path = "malicious/"
    for root, dirs, files in os.walk(phishing_path):
        files = [f for f in files if not f[0] == "."]
        for f in files:
            with open(os.path.join(root, f)) as infile:
                for item in infile.readlines():
                    # Safeguard to prevent adding duplicate data to training set.
                    if item not in training_data:
                        training_data[item.strip('\n')] = 1

    print("[+] Completed.")
    print("\t - Not phishing domains: {}".format(sum(x == 0 for x in training_data.values())))
    print("\t - Phishing domains: {}".format(sum(x == 1 for x in training_data.values())))
    return training_data


#training_data = load_training_data()
#print(training_data)