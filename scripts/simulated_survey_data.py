import numpy as np
import pandas as pd
import re

np.random.seed(202209)
n = 50000

countries = ["France (FR)", "Belgium (BE)", "Spain (ES)", "Portugal (PT)",
             "Germany (DE)", "Italy (IT)", "Netherlands (NL)", "Austria (AT)",
             "Poland (PO)", "Sweden (SE)"]
             
df = pd.DataFrame({
    "gender": np.random.choice(["Male", "Female", "Other", "Don't want to say"], size=n),
    "age": np.random.randint(18, 90, size=n),
    "country": np.random.choice(countries, size=n),
    "city_rural": np.random.choice(["City", "Suburbs", "Rural area"], size=n),
    "edu": np.random.choice(["High school", "University with Bachelor degree", 
                             "University with Master degree", "No School"], size=n),
    "employment_status": np.random.choice(["I'm a student (full-time)", "I'm a student (part-time)",
                                           "I'm employed (full-time)", "I'm employed (part-time)",
                                           "I'm self-employed", "I'm unemployed",
                                           "I am a business owner"], size=n),
    "owns_cat": np.random.choice(["I don't own a cat", "I have one cat", "I have two or more cats"], size=n),
    "likes_dogs": np.random.choice(["Yes", "No"], size=n),
    "household_type": np.random.choice(["I live alone", "I live in a shared flat", 
                                        "I live with my partner", "I live with my family"], size=n),
    "has_kids": np.random.choice(["No Kids", "1 Kid", "2 Kids", "3 or more kids"], size=n),
    "home_type": np.random.choice(["I rent a flat", "I rent a house", "I own a flat", "I own a house"], size=n),
    "has_garden": np.random.choice(["There is no garden",
                                    "There is only a small terrace or balcony",
                                    "There is a small garden (less than 20sqm)", 
                                    "There is a large garden (bigger than 20sqm)"], size=n)
})


import string
LETTERS = string.ascii_uppercase
countries = ["FR", "DE", "IT", "ES", "BE", "NL", "AT", "PT", "PO", "SE"]
# add candidate questions
candidates = [f"candidate {x} ({country})" for x in LETTERS[0:10] for country in countries ]
cols = {}
for candidate in candidates:
    cd = candidate.replace(" ", "_").replace("(","").replace(")","")
    cols[f"q_knows_{cd}"] = np.random.choice([None, f"I have heard of {candidate}", 
                                           "I don't know who that is"], size=n)
    cols[f"q_vote_for_{cd}"] = np.random.choice([None, f"I consider to vote for {candidate}", 
                                             f"I will not vote for {candidate}"], size=n)
    cols[f"q_agrees_with_{cd}"] = np.random.choice([None, f"I strongly disagree with {candidate}",
                                                f"I disagree with {candidate}",
                                                "I neither agree nor disagree",
                                                f"I agree with {candidate}",
                                                f"I strongly agree with {candidate}"], size=n)
                                                
parties = [f"party {x} ({country})" for x in ["Left", "Center", "Right"] for country in countries ]                  
for party in parties:
  prty = party.replace(" ", "_").replace("(","").replace(")","")
  cols[f"q_voted_{prty}_before"] = np.random.choice([None, f"I have voted {party} once or twice before",
                                                  f"I have always voted {party} in the past",
                                                  f"I have never voted {party} before"], size=n)
  cols[f"q_agrees_with_{prty}"] = np.random.choice([None, f"I strongly disagree with {party}",
                                            f"I disagree with {party}",
                                            "I neither agree nor disagree",
                                            f"I agree with {party}",
                                            f"I strongly agree with {party}"], size=n)
  

df = pd.concat([df, pd.DataFrame(cols)], axis=1)
