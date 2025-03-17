from conllu import parse_incr

#######################################
# ЧТЕНИЕ .conllu ФАЙЛА NERUS
#######################################

def read_nerus_conllu_limited(file_path, max_samples=10000):
    sentences = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        # parse_incr читает по одному предложению за раз
        for i, tokenlist in enumerate(parse_incr(f), start=1):
            tokens = []
            tags = []
            for token in tokenlist:
                form = token["form"]
                misc = token["misc"]
                
                if not misc:
                    ner_tag = "O"
                else:
                    ner_tag = misc.get("Tag", "O")
                
                tokens.append(form)
                tags.append(ner_tag)
            
            if tokens:
                sentences.append((tokens, tags))
            
            if max_samples != 0 and i >= max_samples:
                break
    
    return sentences
