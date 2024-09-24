PRED_DICT = {}

#PRED存放的是谓词类
class Predicate:

    def __init__(self, name, var_types):
        """

        :param name:
            string
        :param var_types:
            list of strings
        """
        self.name = name#关系名
        self.var_types = var_types#关系类型
        self.num_args = len(var_types)#几元关系

    def __repr__(self):
        return '%s(%s)' % (self.name, ','.join(self.var_types))
