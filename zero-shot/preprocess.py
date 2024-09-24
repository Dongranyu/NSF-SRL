import re
from common.predicate import Predicate, PRED_DICT
from common.constants import TYPE_SET, const_dict, Fact
from common.formula import Atom, Formula
from os.path import join as joinpath
from os.path import isfile
from common.utils import iterline
from common.cmd_args import cmd_args


def preprocess(ppath, fpath, rpath):
    """

    :param ppath:
        predicate file path#relation file in our code
    :param fpath:
        facts file path
    :param rpath:
        rule file path
    :param qpath:
        query file path

    :return:

    """
    import pdb
    const_path = '/data/ydr2021/image_classification_CUB/AttentionZSL/data/CUB/mln/entities.txt'
    for line in iterline(const_path):
      
        const_dict.add_const('type', line)
   
    
    #assert all(map(isfile, [ppath, fpath, rpath]))

    strip_items = lambda ls: list(map(lambda x: x.strip(), ls))

    pred_reg = re.compile(r'(.*)\((.*)\)')
    pred_id2name = []
    with open(ppath) as f:
        for line in f:

            # skip empty lines
            if line.strip() == '':
                continue

            m = pred_reg.match(line.strip())
            assert m is not None, 'matching predicate failed for %s' % line

            name, var_types = m.group(1), m.group(2)
            var_types = list(map(lambda x: x.strip(), var_types.split(',')))
            pred_id2name.append(name)
            PRED_DICT[name] = Predicate(name, var_types)
            TYPE_SET.update(var_types)
    
    fact_ls = []
  
    for line in iterline(fpath):
        parts = line.split(' ')

        pred_name, e = parts
 
        #assert const_dict.has_const('type', e) 
        #assert pred_name in PRED_DICT

        fact_ls.append(Fact(pred_name, [e], 1))
    
    
    rule_ls = []
    first_atom_reg = re.compile(r'([\d.]+) (!?)([\w\d\W]+)\((.*)\)')
    atom_reg = re.compile(r'(!?)([\w\d\W]+)\((.*)\)')
    with open(rpath,encoding='utf_8') as f:
        for line in f:
        
            
            atom_str_ls = strip_items(line.strip().split(' v '))
            assert len(atom_str_ls) > 1, 'rule length must be greater than 1, but get %s' % line

            atom_ls = []
            rule_weight = 0.0
            for i, atom_str in enumerate(atom_str_ls):
                if i == 0:
                    
                    m = first_atom_reg.match(atom_str)
                    
                    assert m is not None, 'matching atom failed for %s' % atom_str
                    rule_weight = float(m.group(1))
                    neg = m.group(2) == '!'
                    pred_name = m.group(3).strip()
                    var_name_ls = strip_items(m.group(4).split(','))
                else:
                    
                    m = atom_reg.match(atom_str)
                    assert m is not None, 'matching atom failed for %s' % atom_str
                    neg = m.group(1) == '!'
                    pred_name = m.group(2).strip()
                    var_name_ls = strip_items(m.group(3).split(','))

                atom = Atom(neg, pred_name, var_name_ls, PRED_DICT[pred_name].var_types)
                atom_ls.append(atom)
                
            rule = Formula(atom_ls, rule_weight)
            rule_ls.append(rule)
    
    # query_ls = []
    # for line in iterline(qpath):
    #     parts = line.split(' ')

    #     pred_name, e = parts

    #     #assert const_dict.has_const('type', e) 
    #     assert pred_name in PRED_DICT

    #     query_ls.append(Fact(pred_name, [e], 1))
       
    
    return fact_ls, rule_ls
