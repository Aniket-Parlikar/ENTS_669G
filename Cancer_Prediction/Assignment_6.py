# -*- coding: utf-8 -*-

"""
Created on Thu Nov 18 10:34:50 2021
The given code is regarding creating a Multiclass Bayesian Classifier 
to detect whether a patient has flu or not(ie:are they well,have cold or contacted flu).

@author: aniket
"""

import numpy as np

data=np.loadtxt('./hwk6.txt')

#Dividing the given data into the train and test dataset
train=data[:150]
test=data[150:]

#Continuous variables
temp=train[:,1]
bact=train[:,2]

#Categorical variables
n_headache=train[:,3].shape[0]
n_cough=train[:,4].shape[0]
n_musc=train[:,5].shape[0]
n_naus=train[:,6].shape[0]
n_flu_shot=train[:,7].shape[0]

#Target variable
flu=train[:,8]

flu0=train[train[:,8]==0]
flu1=train[train[:,8]==1]
flu2=train[train[:,8]==2]

#Storing variables 
n_flu0=flu0.shape[0]
n_flu1=flu1.shape[0]
n_flu2=flu2.shape[0]
n=train.shape[0]

p_flu0=n_flu0/n
p_flu1=n_flu1/n
p_flu2=n_flu2/n

p_head_flu=np.zeros((3,3))
p_cough_flu=np.zeros((2,3))
p_muscle_flu=np.zeros((2,3))
p_naus_flu=np.zeros((2,3))
p_flushot_flu=np.zeros((2,3))

#Storing the conditional probabilities for the headache variable
n_head0_flu0=flu0[flu0[:,3]==0][:,3].shape[0]
p_head_flu[0,0]=n_head0_flu0/n_flu0

n_head1_flu0=flu0[flu0[:,3]==1][:,3].shape[0]
p_head_flu[1,0]=n_head1_flu0/n_flu0

n_head2_flu0=flu0[flu0[:,3]==2][:,3].shape[0]
p_head_flu[2,0]=n_head2_flu0/n_flu0

n_head0_flu1=flu1[flu1[:,3]==0][:,3].shape[0]
p_head_flu[0,1]=n_head0_flu1/n_flu1

n_head1_flu1=flu1[flu1[:,3]==1][:,3].shape[0]
p_head_flu[1,1]=n_head1_flu1/n_flu1

n_head2_flu1=flu1[flu1[:,3]==2][:,3].shape[0]
p_head_flu[2,1]=n_head2_flu1/n_flu1

n_head0_flu2=flu2[flu2[:,3]==0][:,3].shape[0]
p_head_flu[0,2]=n_head0_flu2/n_flu2

n_head1_flu2=flu2[flu2[:,3]==1][:,3].shape[0]
p_head_flu[1,2]=n_head1_flu2/n_flu2

n_head2_flu2=flu2[flu2[:,3]==2][:,3].shape[0]
p_head_flu[2,2]=n_head2_flu2/n_flu2

#Storing the conditional probabilities for the cough variable
n_cough0_flu0=flu0[flu0[:,4]==0][:,4].shape[0]
#p_cough0_flu0=n_cough0_flu0/n_flu0
p_cough_flu[0,0]=n_cough0_flu0/n_flu0

n_cough1_flu0=flu0[flu0[:,4]==1][:,4].shape[0]
p_cough_flu[1,0]=n_cough1_flu0/n_flu0

n_cough0_flu1=flu1[flu1[:,4]==0][:,4].shape[0]
p_cough_flu[0,1]=n_cough0_flu1/n_flu1

n_cough1_flu1=flu1[flu1[:,4]==1][:,4].shape[0]
p_cough_flu[1,1]=n_cough1_flu1/n_flu1

n_cough0_flu2=flu2[flu2[:,4]==0][:,4].shape[0]
p_cough_flu[0,2]=n_cough0_flu2/n_flu2

n_cough1_flu2=flu2[flu2[:,4]==1][:,4].shape[0]
p_cough_flu[1,2]=n_cough1_flu2/n_flu2

#Storing the conditional probabilities for the muscle ache variable
n_muscle0_flu0=flu0[flu0[:,5]==0][:,5].shape[0]
p_muscle_flu[0,0]=n_muscle0_flu0/n_flu0

n_muscle1_flu0=flu0[flu0[:,5]==1][:,5].shape[0]
p_muscle_flu[1,0]=n_muscle1_flu0/n_flu0

n_muscle0_flu1=flu1[flu1[:,5]==0][:,5].shape[0]
p_muscle_flu[0,1]=n_muscle0_flu1/n_flu0

n_muscle1_flu1=flu1[flu1[:,5]==1][:,5].shape[0]
p_muscle_flu[1,1]=n_muscle1_flu1/n_flu1

n_muscle0_flu2=flu2[flu2[:,5]==0][:,5].shape[0]
p_muscle_flu[0,2]=n_muscle0_flu2/n_flu2

n_muscle1_flu2=flu2[flu2[:,5]==1][:,5].shape[0]
p_muscle_flu[1,2]=n_muscle1_flu2/n_flu2



#Storing the conditional probabilities for the nauseau variable
n_naus0_flu0=flu0[flu0[:,6]==0][:,6].shape[0]
p_naus_flu[0,0]=n_naus0_flu0/n_flu0

n_naus1_flu0=flu0[flu0[:,6]==1][:,6].shape[0]
p_naus_flu[1,0]=n_naus1_flu0/n_flu0

n_naus0_flu1=flu1[flu1[:,6]==0][:,6].shape[0]
p_naus_flu[0,1]=n_naus0_flu1/n_flu1

n_naus1_flu1=flu1[flu1[:,6]==1][:,6].shape[0]
p_naus_flu[1,1]=n_naus1_flu1/n_flu1

n_naus0_flu2=flu2[flu2[:,6]==0][:,6].shape[0]
p_naus_flu[0,2]=n_naus0_flu2/n_flu2

n_naus1_flu2=flu2[flu2[:,6]==1][:,6].shape[0]
p_naus_flu[1,2]=n_naus1_flu2/n_flu2

#Storing the conditional probability for the flu_shot variable
n_flushot0_flu0=flu0[flu0[:,7]==0][:,7].shape[0]
p_flushot_flu[0,0]=n_flushot0_flu0/n_flu0

n_flushot1_flu0=flu0[flu0[:,7]==1][:,7].shape[0]
p_flushot_flu[1,0]=n_flushot1_flu0/n_flu0

n_flushot0_flu1=flu1[flu1[:,7]==0][:,7].shape[0]
p_flushot_flu[0,1]=n_flushot0_flu1/n_flu1

n_flushot1_flu1=flu1[flu1[:,7]==1][:,7].shape[0]
p_flushot_flu[1,1]=n_flushot1_flu1/n_flu1

n_flushot0_flu2=flu2[flu2[:,7]==0][:,7].shape[0]
p_flushot_flu[0,2]=n_flushot0_flu2/n_flu2

n_flushot1_flu2=flu2[flu2[:,7]==1][:,7].shape[0]
p_flushot_flu[1,2]=n_flushot1_flu2/n_flu2

mu_temp_flu=np.zeros(3)
mu_bact_flu=np.zeros(3)

sigma_temp_flu=np.zeros(3)
sigma_bact_flu=np.zeros(3)

#Computing the mean and std deviation for temperature variable
mu_temp_flu[0]=np.mean(flu0[:,1])
sigma_temp_flu[0]=np.std(flu0[:,1])
mu_temp_flu[1]=np.mean(flu1[:,1])
sigma_temp_flu[1]=np.std(flu1[:,1])
mu_temp_flu[2]=np.mean(flu2[:,1])
sigma_temp_flu[2]=np.std(flu2[:,1])

mu_bact_flu[0]=np.mean(flu0[:,2])
sigma_bact_flu[0]=np.std(flu0[:,2])
mu_bact_flu[1]=np.mean(flu1[:,2])
sigma_bact_flu[1]=np.std(flu1[:,2])
mu_bact_flu[2]=np.mean(flu2[:,2])
sigma_bact_flu[2]=np.std(flu2[:,2])

mu_bact=np.mean(bact)
sigma_temp=np.std(bact)

def gaussian_prob(x,sigma,mu):
    exp_part=np.exp(-((x-mu)**2)/(2*(sigma)**2))
    denom= (np.sqrt(2*np.pi)*sigma)
    return exp_part/denom

def predict_proba(data):
    
    '''Function to forecast if a patient wil contct flu or not'''
    predictions=np.array([])
    
    for row in data:
       row=row.flatten()
       temp,bact,head,cough,musc,naus,flushot=row[1:8]
       
       #Calculating the conditional probabilities with the temperauture variable
       p_temp_flu0=gaussian_prob(temp,sigma_temp_flu[0],mu_temp_flu[0])
       p_temp_flu1=gaussian_prob(temp,sigma_temp_flu[1],mu_temp_flu[1])
       p_temp_flu2=gaussian_prob(temp,sigma_temp_flu[2],mu_temp_flu[2])

       #Calculating the condtional probability with the bacteria variable
       p_bact_flu0=gaussian_prob(bact,sigma_bact_flu[0],mu_bact_flu[0])
       p_bact_flu1=gaussian_prob(bact,sigma_bact_flu[1],mu_bact_flu[1])
       p_bact_flu2=gaussian_prob(bact,sigma_bact_flu[2],mu_bact_flu[2])
       
       #Calculating the overall probability that the patient is well
       pred_flu0=p_temp_flu0*p_bact_flu0*p_head_flu[int(head),0]*p_cough_flu[int(cough),0]*p_muscle_flu[int(musc),0]*p_naus_flu[int(naus),0]*p_flushot_flu[int(flushot),0]*p_flu0
       
       #Calculating the overall probablity that the patient is cold
       pred_flu1=p_temp_flu1*p_bact_flu1*p_head_flu[int(head),1]*p_cough_flu[int(cough),1]*p_muscle_flu[int(musc),1]*p_naus_flu[int(naus),1]*p_flushot_flu[int(flushot),1]*p_flu1
       
       #Calculating the overall probability that the patient has flu
       pred_flu2=p_temp_flu2*p_bact_flu2*p_head_flu[int(head),2]*p_cough_flu[int(cough),2]*p_muscle_flu[int(musc),2]*p_naus_flu[int(naus),2]*p_flushot_flu[int(flushot),2]*p_flu2
       
       pred=max(pred_flu0,pred_flu1,pred_flu2)
       #print("p_flu0",p_flu0," p_flu1",p_flu1," p_flu2",p_flu2)
       if pred==pred_flu0:     
           p=0
       elif pred==pred_flu1:
           p=1
       else:
           p=2
       predictions=np.append(predictions,p)
        
    return predictions

actual=test[:,8]
predictions=predict_proba(test)

print("The accuracy of the predicted test data is",np.mean(actual==predictions))
print("The number of correct classifications are",np.sum(actual==predictions))
print("The number of incorrect classifications are",np.sum(actual!=predictions))