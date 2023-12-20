import torch
import numpy as np
import pandas as pd        

def sampler_preterm_birth_age(labels):
    total = len(labels)
    frac_0 = total / np.sum(labels[:]>=37)

    weights = np.ones(len(labels)) * frac_0

    frac_1  = total/ np.sum(labels[:]<32)
    frac_2 = total/ (np.sum(labels[:]<37) - np.sum(labels[:]<=32))
    weights[np.where(labels[:]<32)] = frac_1
    weights[np.where(np.logical_and(labels[:]<37 , labels[:]>=32))] = frac_2
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    print('Term {}'.format(round(frac_0,2)),'preterm {}'.format(round(frac_2,2)),'very preterm {}'.format(round(frac_1,2)))
    return sampler     

def sampler_preterm_scan_age(labels):
    total = len(labels)
    frac_0 = total / np.sum(labels[:]>=37)
    weights = np.ones(len(labels)) * frac_0
    frac_1  = total/ np.sum(labels[:]<37)
    weights[np.where(labels[:]<37)] = frac_1
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    print('Term {}'.format(round(frac_0,2)), 'preterm {}'.format(round(frac_1,2)))

    return sampler     

def sampler_UKB_scan_age(labels):
    total = len(labels)
    frac_0 = total / np.sum(labels[:]>=75) ## before I was dividing that by 2

    weights = np.ones(len(labels)) * frac_0

    frac_1  = (total/ np.sum(labels[:]<55))  #before i was diving by 2
    frac_2 = total/ (np.sum(labels[:]<75) - np.sum(labels[:]<=55))
    weights[np.where(labels[:]<55)] = frac_1
    weights[np.where(np.logical_and(labels[:]<75 , labels[:]>=55))] = frac_2
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    print('Above 75 {}'.format(round(frac_0,2)), 'below 55 {}'.format(round(frac_1,2)), 'between 55 and 75 {}'.format(round(frac_2,2)))

    return sampler     


def sampler_sex_classification(labels):
    total = len(labels)
    frac_0 = total / np.sum(labels[:]==0)
    weights = np.ones(len(labels)) * frac_0
    frac_1  = total/ np.sum(labels[:]==1)
    weights[np.where(labels[:]==1)] = frac_1
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    print('Label 0 {}'.format(round(frac_0,2)),'label 1 {}'.format(round(frac_1,2)))

    return sampler     


def sampler_cardiac_class(labels):
    total = len(labels)
    frac_0 = total / np.sum(labels[:]==0)
    weights = np.ones(len(labels)) * frac_0
    frac_1  = total/ np.sum(labels[:]==1)
    weights[np.where(labels[:]==1)] = frac_1
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return sampler     


def sampler_fi(labels):
    total = len(labels)
    frac_0 = total / np.sum(labels[:]>=20)

    weights = np.ones(len(labels)) * frac_0

    frac_1  = total/ np.sum(labels[:]<10)
    frac_2 = total/ (np.sum(labels[:]<20) - np.sum(labels[:]<=15))
    frac_3 = total/ (np.sum(labels[:]<15) - np.sum(labels[:]<=10))

    weights[np.where(labels[:]<10)] = frac_1
    weights[np.where(np.logical_and(labels[:]<20 , labels[:]>=15))] = frac_2
    weights[np.where(np.logical_and(labels[:]<15 , labels[:]>=10))] = frac_3
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    return sampler    

def sampler_HCP_fluid_intelligence(labels):
    total = len(labels)
    frac_0 = total / np.sum(labels[:]>20)

    weights = np.ones(len(labels)) * frac_0

    frac_1  = total/ np.sum(labels[:]<=17)

    weights[np.where(labels[:]<=17)] = frac_1
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    return sampler    

def new_sampler_HCP_fluid_intelligence(labels):
    total = len(labels)
    frac_0 = total / np.sum(labels[:]>=19)
    weights = np.ones(len(labels)) * frac_0
    frac_1  = total/ np.sum(labels[:]<13)
    frac_2 = total/ (np.sum(labels[:]<19) - np.sum(labels[:]<=13))
    weights[np.where(labels[:]<13)] = frac_1
    weights[np.where(np.logical_and(labels[:]<19 , labels[:]>=13))] = frac_2
    print('Below 13 {}'.format(round(frac_1,2)),'above 19 {}'.format(round(frac_0,2)), 'between 13 and 19 {}'.format(round(frac_2,2)))
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return sampler    

def sampler_q_chat(labels):
    total = len(labels)
    frac_0 = total / np.sum(labels[:]>=40)

    weights = np.ones(len(labels)) * frac_0

    frac_1  = total/ np.sum(labels[:]<20)
    frac_2 = total/ (np.sum(labels[:]<40) - np.sum(labels[:]<=20))
    weights[np.where(labels[:]<20)] = frac_1
    weights[np.where(np.logical_and(labels[:]<40 , labels[:]>=20))] = frac_2
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    return sampler     

def sampler_q_chat_class(labels):
    total = len(labels)
    frac_0 = total / np.sum(labels[:]==0)
    weights = np.ones(len(labels)) * frac_0
    frac_1  = total/ np.sum(labels[:]==1)
    weights[np.where(labels[:]==1)] = frac_1
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return sampler    


def balance_dataset(df):

    # Split the dataframe based on the 'is_movie' column
    df_1 = df[df['labels'] == 1]
    df_0 = df[df['labels'] == 0]

    # Determine the minimum count between the two dataframes
    min_count = min(len(df_1), len(df_0))

    # Sample the minimum count from each dataframe
    sampled_df_1 = df_1.sample(min_count)
    sampled_df_0 = df_0.sample(min_count)

    # Concatenate the two sampled dataframes
    return pd.concat([sampled_df_1, sampled_df_0], axis=0).reset_index(drop=True)
