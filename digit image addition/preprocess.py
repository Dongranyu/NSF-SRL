import re
from common.predicate import Predicate,PRED_DICT
from common.constants import const_dict,TYPE_SET,Fact
from common.formula import Formula,Atom
import pdb
def data_process(ppath,fpath,cpath,rpath):

    strip_items = lambda ls: list(map(lambda x: x.strip(), ls))
    pred_reg = re.compile(r'(.*)\((.*)\)')
    pred_id2name = []
    with open(ppath) as f:
        for line in f:
            m = pred_reg.match(line.strip())
            assert m is not None, 'matching predicate failed for %s' % line

            name, var_types = m.group(1), m.group(2)
            var_types = list(map(lambda x: x.strip(), var_types.split(',')))
           
            pred_id2name.append(name)
            PRED_DICT[name] = Predicate(name, var_types)
            TYPE_SET.update(var_types)
        
    fact_ls = []
    with open(fpath) as f:
        for line in f:#这里的name都是二元谓词
            val = 1
            name = pred_id2name[int(line.strip().split(' ')[1])]
            consts = [line.strip().split()[0],line.strip().split()[2]]
            fact_ls.append(Fact(name,consts,val))
            for var_type in PRED_DICT[name].var_types:
                const_dict.add_const(var_type, consts.pop(0))
    with open(cpath) as f:
        for line in f:
            val = 1
            name = pred_id2name[int(line.strip().split(' ')[1])]
            const = [line.strip().split()[0]]
            fact_ls.append(Fact(name,const,val))

    rule_ls = []
    first_atom_reg = re.compile(r'([\d.]+) (!?)(.*)\((.*)\)')
    atom_reg = re.compile(r'(!?)(.*)\((.*)\)')
    #with open(rpath,encoding='gbk') as f:
    with open(rpath,encoding='utf_8') as f:
        for line in f:

            atom_str_ls = strip_items(line.strip().split('∨'))
            atom_ls = []
            rule_weight = 0.0
            for i, atom_str in enumerate(atom_str_ls):
                if i == 0:
                    m = first_atom_reg.match(atom_str)
                    assert m is not None, 'matching atom failed for %s' % atom_str
                    rule_weight = float(m.group(1))
                    neg = m.group(2) == '!'
                    pred_name = m.group(3).strip()
                    if len(m.group(4).split(','))<2:
                        var_name_ls = [m.group(4)]
                    else:
                        var_name_ls = strip_items(m.group(4).split(','))
                else:
                    m = atom_reg.match(atom_str)
                    assert m is not None, 'matching atom failed for %s' % atom_str
                    neg = m.group(1) == '!'
                    pred_name = m.group(2).strip()
                    if len(m.group(3).split(','))<2:
                        var_name_ls = [m.group(3)]
                    else:
                        var_name_ls = strip_items(m.group(3).split(','))
                    # var_name_ls = strip_items(m.group(3).split(','))
                if pred_name not in PRED_DICT:
                    atom = Atom(neg, pred_name, var_name_ls, ['type'])
                    atom_ls.append(atom)
                else:
                    atom = Atom(neg, pred_name, var_name_ls, PRED_DICT[pred_name].var_types)
                    atom_ls.append(atom)
            rule = Formula(atom_ls, rule_weight)
            rule_ls.append(rule)

    return fact_ls,rule_ls,pred_id2name

if __name__ == "__main__":
    fact,rule,pred = data_process("/home/yudongran/ydr2/general_framework/data/MNIST/mln_data/relations.txt",\
                             "/home/yudongran/ydr2/general_framework/data/MNIST/mln_data/fact.txt","/home/yudongran/ydr2/general_framework/data/MNIST/mln_data/rules.txt")
    print(rule)