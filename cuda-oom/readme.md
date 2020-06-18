<table>
<tr><td colspan="5" style="text-align: center;"> ('cudnn conv2d', 'True') </td></tr>
<tr><td colspan="2" style="text-align: center;"> clean_in_exception True </td><td style="text-align: center;">---</td><td colspan="1" style="text-align: center;"> clean_in_exception False </td></tr>
<tr><td style="text-align: center;">req_grad</td><td style="text-align: center;">oom size</td><td style="text-align: center;">---</td><td style="text-align: center;">oom size</td></tr>
<tr><td>True</td><td>5255 4340 4340 4340</td><td>---</td><td>5255 4340 4340 4340</td></tr>
<tr><td>False</td><td>7690 6995 6995 6995</td><td>---</td><td>7690 6995 6995 6995</td></tr>
</table>
<table>
<tr><td colspan="5" style="text-align: center;"> ('cudnn conv2d', 'False') </td></tr>
<tr><td colspan="2" style="text-align: center;"> clean_in_exception True </td><td style="text-align: center;">---</td><td colspan="1" style="text-align: center;"> clean_in_exception False </td></tr>
<tr><td style="text-align: center;">req_grad</td><td style="text-align: center;">oom size</td><td style="text-align: center;">---</td><td style="text-align: center;">oom size</td></tr>
<tr><td>True</td><td>4775 4340 4340 4340</td><td>---</td><td>4775 3945 3945 3945</td></tr>
<tr><td>False</td><td>5780 5255 5255 5255</td><td>---</td><td>5780 4340 4340 4340</td></tr>
</table>
<table>
<tr><td colspan="5" style="text-align: center;"> ('cuda kernel without cudnn, may have internal tensor', 'True') </td></tr>
<tr><td colspan="2" style="text-align: center;"> clean_in_exception True </td><td style="text-align: center;">---</td><td colspan="1" style="text-align: center;"> clean_in_exception False </td></tr>
<tr><td style="text-align: center;">req_grad</td><td style="text-align: center;">oom size</td><td style="text-align: center;">---</td><td style="text-align: center;">oom size</td></tr>
<tr><td>True</td><td>6995 6355 6355 6355</td><td>---</td><td>6995 6355 6355 6355</td></tr>
<tr><td>False</td><td>9310 9310 9310 9310</td><td>---</td><td>9310 8460 8460 8460</td></tr>
</table>
<table>
<tr><td colspan="5" style="text-align: center;"> ('cuda kernel without cudnn, may have internal tensor', 'False') </td></tr>
<tr><td colspan="2" style="text-align: center;"> clean_in_exception True </td><td style="text-align: center;">---</td><td colspan="1" style="text-align: center;"> clean_in_exception False </td></tr>
<tr><td style="text-align: center;">req_grad</td><td style="text-align: center;">oom size</td><td style="text-align: center;">---</td><td style="text-align: center;">oom size</td></tr>
<tr><td>True</td><td>5780 5255 5780 5255</td><td>---</td><td>5780 5255 5255 5255</td></tr>
<tr><td>False</td><td>7690 7690 7690 7690</td><td>---</td><td>7690 7690 7690 7690</td></tr>
</table>
<table>
<tr><td colspan="5" style="text-align: center;"> ('aten native, no weight', 'True') </td></tr>
<tr><td colspan="2" style="text-align: center;"> clean_in_exception True </td><td style="text-align: center;">---</td><td colspan="1" style="text-align: center;"> clean_in_exception False </td></tr>
<tr><td style="text-align: center;">req_grad</td><td style="text-align: center;">oom size</td><td style="text-align: center;">---</td><td style="text-align: center;">oom size</td></tr>
<tr><td>True</td><td>5255 5255 5255 5255</td><td>---</td><td>5255 5255 5255 5255</td></tr>
<tr><td>False</td><td>7690 7690 7690 7690</td><td>---</td><td>7690 6995 6995 6995</td></tr>
</table>
<table>
<tr><td colspan="5" style="text-align: center;"> ('aten native, no weight', 'False') </td></tr>
<tr><td colspan="2" style="text-align: center;"> clean_in_exception True </td><td style="text-align: center;">---</td><td colspan="1" style="text-align: center;"> clean_in_exception False </td></tr>
<tr><td style="text-align: center;">req_grad</td><td style="text-align: center;">oom size</td><td style="text-align: center;">---</td><td style="text-align: center;">oom size</td></tr>
<tr><td>True</td><td>4775 4775 4775 4775</td><td>---</td><td>4775 3945 3945 3945</td></tr>
<tr><td>False</td><td>6355 6355 6355 6355</td><td>---</td><td>6355 6355 6355 6355</td></tr>
</table>
<table>
<tr><td colspan="5" style="text-align: center;"> ('nn, no weight', 'True') </td></tr>
<tr><td colspan="2" style="text-align: center;"> clean_in_exception True </td><td style="text-align: center;">---</td><td colspan="1" style="text-align: center;"> clean_in_exception False </td></tr>
<tr><td style="text-align: center;">req_grad</td><td style="text-align: center;">oom size</td><td style="text-align: center;">---</td><td style="text-align: center;">oom size</td></tr>
<tr><td>True</td><td>5255 5255 5255 5255</td><td>---</td><td>5255 5255 5255 5255</td></tr>
<tr><td>False</td><td>7690 6995 6995 6995</td><td>---</td><td>7690 6995 6995 6995</td></tr>
</table>
<table>
<tr><td colspan="5" style="text-align: center;"> ('nn, no weight', 'False') </td></tr>
<tr><td colspan="2" style="text-align: center;"> clean_in_exception True </td><td style="text-align: center;">---</td><td colspan="1" style="text-align: center;"> clean_in_exception False </td></tr>
<tr><td style="text-align: center;">req_grad</td><td style="text-align: center;">oom size</td><td style="text-align: center;">---</td><td style="text-align: center;">oom size</td></tr>
<tr><td>True</td><td>4775 4340 4775 4340</td><td>---</td><td>4775 3945 3945 3945</td></tr>
<tr><td>False</td><td>6355 6355 6355 6355</td><td>---</td><td>6355 6355 6355 6355</td></tr>
</table>
