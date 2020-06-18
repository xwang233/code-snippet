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
    data = {}
    for key in d:
        k_new = key[:2]

        if k_new not in tables:
            tk: list = tables[k_new]
            tk.append('<table>\n')
            tk.append(
                '<tr>'
                f'<td colspan="5"> op={k_new[0]}; set_to_none={k_new[1]} </td>'
                '</tr>\n'
            )
            tk.append(
                '<tr>'
                '<td colspan="2"> clean_in_exception True </td>'
                '<td colspan="1"> clean_in_exception False </td>'
                '</tr>\n'
            )
            tk.append(
                '<tr>'
                '<td>req_grad</td><td>oom size</td>'
                '<td>oom size</td>'
                '</tr>\n'
            )

    with open(fn_md, 'w') as f_md:
        for k_new in tables:
            f_md.writelines(tables[k_new])

            f_md.write(
                '<tr>'
                '<td>True</td><td>' +
                ' '.join(d[(*k_new, 'True', 'True')]) +
                '</td><td>' +
                ' '.join(d[(*k_new, 'False', 'True')]) +
                '</td></tr>\n'
            )
            f_md.write(
                '<tr>'
                '<td>False</td><td>' + 
                ' '.join(d[(*k_new, 'True', 'False')]) +
                '</td><td>' +
                ' '.join(d[(*k_new, 'False', 'False')]) +
                '</td></tr>\n'
            )

            f_md.write('</table>\n')

if __name__ == "__main__":
    main('results.txt', 'readme.md')