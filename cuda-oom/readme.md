('cudnn conv2d', 'True', 'True')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>5255 4340 4340 4340 </td></tr>
<tr><td>False</td><td>7690 6995 6995 6995 </td></tr>
</table>

('cudnn conv2d', 'True', 'False')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>5255 4340 4340 4340 </td></tr>
<tr><td>False</td><td>7690 6995 6995 6995 </td></tr>
</table>

('cudnn conv2d', 'False', 'True')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>4775 4340 4340 4340 </td></tr>
<tr><td>False</td><td>5780 5255 5255 5255 </td></tr>
</table>

('cudnn conv2d', 'False', 'False')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>4775 3945 3945 3945 </td></tr>
<tr><td>False</td><td>5780 4340 4340 4340 </td></tr>
</table>

('cuda kernel without cudnn, may have internal tensor', 'True', 'True')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>6995 6355 6355 6355 </td></tr>
<tr><td>False</td><td>9310 9310 9310 9310 </td></tr>
</table>

('cuda kernel without cudnn, may have internal tensor', 'True', 'False')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>6995 6355 6355 6355 </td></tr>
<tr><td>False</td><td>9310 8460 8460 8460 </td></tr>
</table>

('cuda kernel without cudnn, may have internal tensor', 'False', 'True')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>5780 5255 5780 5255 </td></tr>
<tr><td>False</td><td>7690 7690 7690 7690 </td></tr>
</table>

('cuda kernel without cudnn, may have internal tensor', 'False', 'False')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>5780 5255 5255 5255 </td></tr>
<tr><td>False</td><td>7690 7690 7690 7690 </td></tr>
</table>

('aten native, no weight', 'True', 'True')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>5255 5255 5255 5255 </td></tr>
<tr><td>False</td><td>7690 7690 7690 7690 </td></tr>
</table>

('aten native, no weight', 'True', 'False')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>5255 5255 5255 5255 </td></tr>
<tr><td>False</td><td>7690 6995 6995 6995 </td></tr>
</table>

('aten native, no weight', 'False', 'True')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>4775 4775 4775 4775 </td></tr>
<tr><td>False</td><td>6355 6355 6355 6355 </td></tr>
</table>

('aten native, no weight', 'False', 'False')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>4775 3945 3945 3945 </td></tr>
<tr><td>False</td><td>6355 6355 6355 6355 </td></tr>
</table>

('nn, no weight', 'True', 'True')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>5255 5255 5255 5255 </td></tr>
<tr><td>False</td><td>7690 6995 6995 6995 </td></tr>
</table>

('nn, no weight', 'True', 'False')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>5255 5255 5255 5255 </td></tr>
<tr><td>False</td><td>7690 6995 6995 6995 </td></tr>
</table>

('nn, no weight', 'False', 'True')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>4775 4340 4775 4340 </td></tr>
<tr><td>False</td><td>6355 6355 6355 6355 </td></tr>
</table>

('nn, no weight', 'False', 'False')
<table>
<tr><td>req_grad</td><td>oom size</td></tr>
<tr><td>True</td><td>4775 3945 3945 3945 </td></tr>
<tr><td>False</td><td>6355 6355 6355 6355 </td></tr>
</table>

