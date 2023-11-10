import pandas as pd
from data.dataset import KWDLC


class PrototypicalNetworkDataLoader:
    def __init__(self):
        self.ds = KWDLC()

    def __iter__(self):
        self.ds.train_data_yes = self.ds.train_data_yes.sample(
            frac=1, ignore_index=True)
        self.ds.train_data_no = self.ds.train_data_no.sample(
            frac=1, ignore_index=True)
        num_of_support = 5
        num_of_query = 45

        nq_yes = (num_of_query - num_of_support) // 2
        nq_no = (num_of_query + num_of_support) // 2

        n_episode = min(len(self.ds.train_data_yes),
                        len(self.ds.train_data_no))//nq_no

        for i in range(n_episode):
            support_set = self.ds.train_data_yes[nq_no *
                                                 i:nq_no*i+num_of_support]
            query_set_yes = self.ds.train_data_yes[nq_no *
                                                   i+num_of_support:nq_no*(i+1)]
            query_set_no = self.ds.train_data_no[nq_no*i:nq_no*(
                i+1)]
            query_set = pd.concat([query_set_yes, query_set_no])
            batch = [support_set, query_set]
            yield batch


# class PrototypicalNetworkDataLoader:
#     def __init__(self):
#         self.ds = KWDLC()

#     def __iter__(self, mode='train'):
#         self.ds.train_data_yes = self.ds.train_data_yes.sample(
#             frac=1, ignore_index=True)
#         self.ds.train_data_no = self.ds.train_data_no.sample(
#             frac=1, ignore_index=True)
#         num_of_support = 5
#         num_of_query = 45
#         nq_yes = (num_of_query - num_of_support) // 2
#         nq_no = (num_of_query + num_of_support) // 2
#         for i in range(len(self.ds.train_data_yes)//nq_no):
#             support_set = self.ds.train_data_yes[nq_no *
#                                                  i:nq_no*i+num_of_support]
#             query_set_yes = self.ds.train_data_yes[nq_no *
#                                                    i+num_of_support:nq_no*(i+1)]
#             query_set_no = self.ds.train_data_no[nq_no*i:nq_no*(i+1)]
#             query_set = pd.concat([query_set_yes, query_set_no])
#             batch = [support_set, query_set]
#             yield batch


# class EvalPrototypicalNetworkDataLoader:
#     def __init__(self, data='Rakuten'):
#         # self.ds = Combine_Rakuten_Chatgpt()
#         self.ds = CombineTrainingData()
#         if data == 'Rakuten':
#             # rakuten_data = OneSentenceTest()
#             rakuten_data = RakutenTest()
#             self.test_data_yes = rakuten_data.test_data_yes
#             self.test_data_no = rakuten_data.test_data_no

#     def get_batch(self, mode='val'):
#         num_of_support = 32
#         self.ds.train_data_yes = self.ds.train_data_yes.sample(
#             frac=1, ignore_index=True)
#         support_set = self.ds.train_data_yes[:num_of_support]
#         if mode == 'val':
#             query_set = pd.concat(
#                 [self.ds.val_data_yes, self.ds.val_data_no])
#         else:
#             query_set = pd.concat(
#                 [self.test_data_yes, self.test_data_no])
#         batch = [support_set, query_set]
#         return batch

class EvalPrototypicalNetworkDataLoader:
    def __init__(self, data='KWDLC'):
        self.ds = KWDLC()

    def get_batch(self, mode='val'):
        num_of_support = 32
        self.ds.train_data_yes = self.ds.train_data_yes.sample(
            frac=1, ignore_index=True)
        support_set = self.ds.train_data_yes[:num_of_support]
        if mode == 'val':
            query_set = pd.concat([self.ds.val_data_yes, self.ds.val_data_no])
        else:
            query_set = pd.concat(
                [self.ds.test_data_yes, self.ds.test_data_no])
        batch = [support_set, query_set]
        return batch
