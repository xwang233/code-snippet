from collections import defaultdict

def main(fn, fn_md):
    with open(fn, 'r') as f:
        lines = f.readlines()
    
    d = {}
    for i in range(0,len(lines),10):
        parse = lambda x: x[x.index(' ') + 1 : ].rstrip()

        op = parse(lines[i+0])
        set_to_none = parse(lines[i+1])
        clean_in_exception = parse(lines[i+2])
        req_grad = parse(lines[i+3])

        oom_sizes = [x.rstrip() for x in lines[i+5:i+9]]

        key = (op, set_to_none, clean_in_exception, req_grad)
        val = [x for x in oom_sizes]

        d[key] = val
    
    # print(d)

    tables = defaultdict(list)
    for key in d:
        k_new = key[:3]

        if k_new not in tables:
            tk: list = tables[k_new]
            tk.append('<table>\n')
            tk.append('<tr><td>req_grad</td><td>oom size</td></tr>\n')
        tk: list = tables[k_new]
        tk.append(f'<tr><td>{key[3]}</td><td>')
        for s in d[key]:
            tk.append(f'{s} ')
        tk.append('</td></tr>\n')
    
    for k_new in tables:
        tables[k_new].append('</table>\n')
    
    with open(fn_md, 'w') as f_md:
        for k_new in tables:
            f_md.write(f'{k_new}\n')
            f_md.writelines(tables[k_new])
            f_md.write('\n')

if __name__ == "__main__":
    main('results.txt', 'readme.md')