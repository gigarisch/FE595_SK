# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:03:58 2019

@author: gordon.garisch
"""

import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


def influencer(X,Y,features):
    '''
    Function that accepts independent variables (X) , a depdenent variable (Y),
    and the name of the independent variables/features (features).
    
    The fucntion then performs linear regression,
    returning the name of the most influential variable, measured as the
    maximum of the absolute values of coefficients, as well as the value of 
    the coefficient.
    '''
        # Created linear regression object
    lr = LinearRegression()
    
    # Fit model to data
    lr.fit(X,Y)
    
    # Return name of top feature and value of that features fitted coefficient
    return (features[abs(lr.coef_).argmax()],
                    lr.coef_[abs(lr.coef_).argmax()])
def main():
    
    # Load boston data
    boston = load_boston()
    
    # Extract independent and dependent variables, X and Y.
    X = boston.data
    Y = boston.target
    # Retrieve name of features
    features = boston.feature_names
    
    # Retrieve top influencer and top influencer coefficient value
    top_infl, coefficient = influencer(X,Y,features)
    
    # Output results
    print('The top influencer on home prices regressing on all independent variables is {}\n with a coefficient of {}.\nThe definition of {} is: {}.\n'.format(top_infl,
          coefficient,
          top_infl,
          re.findall(re.escape(top_infl)+'(.+)',boston.DESCR)[0].strip())) 
    
    # Extra step #1

    # Remove top influencer from data to retrieve second influencer
    X=np.delete(X,np.where(features==top_infl),1)
    features=np.delete(features,np.where(features==top_infl),0)
    
    # Retrieve top influencer and top influencer coefficient value
    top_infl2, coefficient = influencer(X,Y,features)
    
    # Output results
    print('The top influencer on home prices regressing on all independent variables, except {}, is {}\n with a coefficient of {}.\nThe definition of {} is: {}.\n'.format(top_infl,top_infl2,
          coefficient,
          top_infl2,
          re.findall(re.escape(top_infl2)+'(.+)',boston.DESCR)[0].strip())) 
    
    
    # Extra step #2

    # Remove top influencer from data to retrieve second influencer
    X=np.delete(X,np.where(features==top_infl2),1)
    features=np.delete(features,np.where(features==top_infl2),0)
    
    # Retrieve top influencer and top influencer coefficient value
    top_infl3, coefficient = influencer(X,Y,features)
    
    # Output results
    print('The top influencer on home prices regressing on all independent variables, except {} and {}, is {}\n with a coefficient of {}.\nThe definition of {} is: {}.\n'.format(top_infl,
          top_infl,
          top_infl2,
          coefficient,
          top_infl3,
          re.findall(re.escape(top_infl3)+'(.+)',boston.DESCR)[0].strip())) 
    
if __name__ == "__main__":
    main()
