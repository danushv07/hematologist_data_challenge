metadata_test=example_metadata.loc[test_index,:]
ace_metadata_test=metadata_test.loc[metadata_test['dataset']=='Ace_20'].reset_index(drop = True)
mat_metadata_test=metadata_test.loc[metadata_test['dataset']=='Mat_19'].reset_index(drop = True)
