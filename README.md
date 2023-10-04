# Climate Variable Downscaling with Conditional Normalizing Flows

Predictions of global climate models typically operate on coarse spatial scales due to the large computational costs of climate simulations. This has led to a considerable interest in methods for statistical downscaling, a similar process to super-resolution in the computer vision context, to provide more local and regional climate information. In this work, we apply conditional normalizing flows to the task of climate variable downscaling. This approach allows for a probabilistic interpretation of the predictions, while also capturing the stochasticity inherent in the relationships among fine and coarse spatial scales. We showcase its successful performance on an ERA5 water content dataset for different upsampling factors. Additionally, we show that the method allows us to assess the predictive uncertainty in terms of standard deviation from the fitted conditional distribution mean.

![high_res_gt_397](https://github.com/christina-winkler/clim-var-ds-cnf/assets/33231216/06e74758-6f62-4e1f-a06c-0c2200a176ee)

![mu_0 5_logstep_397_test](https://github.com/christina-winkler/clim-var-ds-cnf/assets/33231216/c0969804-0cc5-470a-8616-eeafcfd8eaf1)


![std_multiplot_35](https://github.com/christina-winkler/clim-var-ds-cnf/assets/33231216/16603f04-023d-439f-be55-2d4a45525d41)
