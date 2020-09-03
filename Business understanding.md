# Conclusion : 
After performing exploratory data analysis and feature selection we found that relevant attribute for predicting the target attributes
are :-
 * "CCAvg" which is monthly credict card spending.
 * "Family" no of family members
 * "CD account"
 * "Avgsaving" which is derived as [ Avgsaving = Income - CCAvg*12  ]
              
	      Correlation of above all mentioned attributes are  relatively higher wrt the target attribute "Personal Loan".
 and also while applying different feature selection criteria those features turned out to be of high interest.
 some conclusions to EDA are as follows :
 * less family member => less likely to take loan
 * if any person doesnot have CD account it is more likely he will not take loan
 * people not taking loan have very left skewed income distribution same is with CCAvg
 * Distribution of avgsavings seems to more approx to normal curve compare to income from which it was initially created.
  
	      Collecting good quality of data is always helpfull for better prediction of target bank must collect more information related to customers savings , spending
  families net income , families net spending or savings  which was observed while data analysis.
 
 
