#!/usr/bin/env python

'''

'''

#Python standard import


#Third party import


#Local import




def print_MC_cov(variables, MC_cor, MC_cov):

    print('')
    hprint('MCMC mean estimator correlation matrix')
    print('')

    header = ''.join(['{:<4} |'.format('')] + [' {:<14} |'.format(var) for var in variables])
    print(header)
    for row in MC_cor:
        pr = ''.join(['{:<4} |'.format(row['variable'])] + [' {:<14.6f} |'.format(row[var]) for var in variables])
        print(pr)

    print('')
    hprint('MCMC mean estimator covariance matrix')
    print('')

    header = ''.join(['{:<4} |'.format('')] + [' {:<14} |'.format(var) for var in variables])
    print(header)
    for row in MC_cov:
        pr = ''.join(['{:<4} |'.format(row['variable'])] + [' {:<14.6f} |'.format(row[var]) for var in variables])
        print(pr)



def print_covariance(results, **kwargs):

    post_cov = results.covariance()

    print('')
    hprint('Posterior covariance matrix')
    print('')

    header = ''.join(['{:<4} |'.format('')] + [' {:<14} |'.format(var) for var in results.variables])
    print(header)
    for row in post_cov:
        pr = ''.join(['{:<4} |'.format(row['variable'])] + [' {:<14.6f} |'.format(row[var]) for var in results.variables])
        print(pr)
