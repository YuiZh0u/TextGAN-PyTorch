def get_keywords(dataset):
    if dataset == 'covid_tweets':
        covid_tweets = ['corona', 'virus', 'coronavirus', 'covid19', 'covid', 'pandemic', 'epidemic', 'lockdown', 'quarantine', 'quarantined', 'infected', 'infection', 'outbreak', 'spread', 'spreading', 'sanitizer', 'contagious', 'immune', 'symptoms', 'doctors', 'patients', 'patient', 'hospital', 'hospitals', 'disease', 'deaths', 'vaccine', 'vaccines', 'viruses', 'cold', 'fever', 'flu', 'cough', 'coughing', 'coughed', 'coughs', 'sick', 'pneumonia', 'sickness', 'medicine', 'hygiene', 'hands', 'hand', 'stay', 'home', 'people', 'ppl', 'distancing', 'wuhan', 'china', 'italy', 'india', 'chinese', 'indian', 'world']
        dataset = covid_tweets
    return dataset