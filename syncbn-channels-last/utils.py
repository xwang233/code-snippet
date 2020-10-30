kw = {
        'batch_norm_stats': [
            'batch_norm_collect_statistics_kernel',
            'batch_norm_collect_statistics_channels_last_kernel',
            'welford_kernel<',
            'welford_kernel_c_last'
        ],
        'batch_norm_elemt': [
            'batch_norm_transform_input_kernel',
            'batch_norm_transform_input_channels_last_kernel',
            'batchnorm_forward_kernel',
            'batchnorm_forward_c_last_kernel'
        ],
        'batch_norm_backward_reduce': [
            'batch_norm_backward_reduce_kernel',
            'batch_norm_backward_reduce_channels_last_kernel',
            'reduce_bn_kernel',
            'reduce_bn_c_last_kernel'
        ],
        'batch_norm_backward_elemt': [
            'batch_norm_backward_elemt_kernel',
            'batch_norm_backward_elemt_channels_last_kernel',
            'batchnorm_backward_kernel',
            'batchnorm_backward_c_last_kernel'
        ]
    }
