def output(test):
    '''                                                                                                                                                                                Return a DF with only the predicted columns                                                                                                                                      
    Params                                                                                                                                                                                  test:      Output test Dataframe                                                                                                                                                 
    Returns
        output:    Output dataframe with only the predicted cols.                                                                                                                        
    '''
    return test[[
            'Return_to_owner',
            'Euthanasia',
            'Adoption',
            'Transfer',
            'Died'
            ]]

def logloss(output):
    '''                                                                                                                                                                              
    Return the logloss evaluator for the given output values                                                                                                                         
                                                                                                                                                                                     
    Params                                                                                                                                                                           
    output:    Output probabilities as a DF.                                                                                                                                         
                                                                                                                                                                                     
    Returns                                                                                                                                                                          
    logloss                                                                                                                                                                          
    '''
    return -(1/nrow(output))*output.sum(axis=1)
