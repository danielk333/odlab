if __name__=='__main__':
    ccsds_file = './data/uhf_test_data/events/2002-009A-1473150428.tdm'
    data = read_ccsds(ccsds_file)
    print(data)