## op = cudnn conv2d
<table>
<tr><td colspan="5"> set_to_none=True </td></tr>
<tr><td colspan="2"> clean_in_exception True </td><td colspan="1"> clean_in_exception False </td></tr>
<tr><td>req_grad</td><td>oom size</td><td>oom size</td></tr>
<tr><td>True</td><td>5255 4340 4340 4340</td><td>5255 4340 4340 4340</td></tr>
<tr><td>False</td><td>7690 6995 6995 6995</td><td>7690 6995 6995 6995</td></tr>
</table>

<table>
<tr><td colspan="5"> set_to_none=False </td></tr>
<tr><td colspan="2"> clean_in_exception True </td><td colspan="1"> clean_in_exception False </td></tr>
<tr><td>req_grad</td><td>oom size</td><td>oom size</td></tr>
<tr><td>True</td><td>4775 4340 4340 4340</td><td>4775 3945 3945 3945</td></tr>
<tr><td>False</td><td>5780 5255 5255 5255</td><td>5780 4340 4340 4340</td></tr>
</table>

----

## op = cuda kernel without cudnn, may have internal tensor
<table>
<tr><td colspan="5"> set_to_none=True </td></tr>
<tr><td colspan="2"> clean_in_exception True </td><td colspan="1"> clean_in_exception False </td></tr>
<tr><td>req_grad</td><td>oom size</td><td>oom size</td></tr>
<tr><td>True</td><td>6995 6355 6355 6355</td><td>6995 6355 6355 6355</td></tr>
<tr><td>False</td><td>9310 9310 9310 9310</td><td>9310 8460 8460 8460</td></tr>
</table>

<table>
<tr><td colspan="5"> set_to_none=False </td></tr>
<tr><td colspan="2"> clean_in_exception True </td><td colspan="1"> clean_in_exception False </td></tr>
<tr><td>req_grad</td><td>oom size</td><td>oom size</td></tr>
<tr><td>True</td><td>5780 5255 5780 5255</td><td>5780 5255 5255 5255</td></tr>
<tr><td>False</td><td>7690 7690 7690 7690</td><td>7690 7690 7690 7690</td></tr>
</table>

----

## op = aten native, no weight
<table>
<tr><td colspan="5"> set_to_none=True </td></tr>
<tr><td colspan="2"> clean_in_exception True </td><td colspan="1"> clean_in_exception False </td></tr>
<tr><td>req_grad</td><td>oom size</td><td>oom size</td></tr>
<tr><td>True</td><td>5255 5255 5255 5255</td><td>5255 5255 5255 5255</td></tr>
<tr><td>False</td><td>7690 7690 7690 7690</td><td>7690 6995 6995 6995</td></tr>
</table>

<table>
<tr><td colspan="5"> set_to_none=False </td></tr>
<tr><td colspan="2"> clean_in_exception True </td><td colspan="1"> clean_in_exception False </td></tr>
<tr><td>req_grad</td><td>oom size</td><td>oom size</td></tr>
<tr><td>True</td><td>4775 4775 4775 4775</td><td>4775 3945 3945 3945</td></tr>
<tr><td>False</td><td>6355 6355 6355 6355</td><td>6355 6355 6355 6355</td></tr>
</table>

----

## op = nn, no weight
<table>
<tr><td colspan="5"> set_to_none=True </td></tr>
<tr><td colspan="2"> clean_in_exception True </td><td colspan="1"> clean_in_exception False </td></tr>
<tr><td>req_grad</td><td>oom size</td><td>oom size</td></tr>
<tr><td>True</td><td>5255 5255 5255 5255</td><td>5255 5255 5255 5255</td></tr>
<tr><td>False</td><td>7690 6995 6995 6995</td><td>7690 6995 6995 6995</td></tr>
</table>

<table>
<tr><td colspan="5"> set_to_none=False </td></tr>
<tr><td colspan="2"> clean_in_exception True </td><td colspan="1"> clean_in_exception False </td></tr>
<tr><td>req_grad</td><td>oom size</td><td>oom size</td></tr>
<tr><td>True</td><td>4775 4340 4775 4340</td><td>4775 3945 3945 3945</td></tr>
<tr><td>False</td><td>6355 6355 6355 6355</td><td>6355 6355 6355 6355</td></tr>
</table>

----

