'''
    * The template file is called All_Experiments_Results_template.txt
    * You should have it before running the main_test {to check the reducability 
      of the code.
    * After running the main_test 'All_Experiments_Results' is generated.
    * check reults compare the two files.
'''
import pandas as pd

def check_results():
    '''Compares the exp file with templated pre_generated results file.
       To compare the reducability of the code.

        Args:
            exp_file: the file which is extracted after running main_test.
    '''

    result1 = pd.read_csv('All_Experiments_Results_template.txt')
    result2 = pd.read_csv('All_Experiments_Results.txt')

    try:
        pd.testing.assert_frame_equal(result1, result2, check_dtype=False)
        return True
    except:
        return False

'''
# The template file 
check_results()
'''
