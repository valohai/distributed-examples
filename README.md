# Valohai Distributed Examples

This repository serves as an example for the [Valohai MLOps platform][vh]. 
It implements various distributed machine learning examples with varying degree of complexity. 

Examples are categorized to directories by used framework (_e.g. pytorch_examples_), sorted
by increasing complexity (_i.e. 00, 01, etc._) and the filename describes communication technologies used
(_e.g. `gloo` or `mpi`_).

[vh]: https://valohai.com/

## Usage

All the scripts are self-contained and intentionally only use dependencies described in 
the related `requirements*.txt` files.

You can use the examples as-is if you already have access to a Valohai private worker or full-private installation.

To tune an example to your own needs and codebase, find the related Valohai `step` 
in the `valohai.yaml` and start building from that.

Main points to look for are:

* Docker image used, and potentially building a new image based on that for speed
* What and how dependencies are installed
* How the machine learning training script is being called
* What extra environmental variables are specified in the YAML
* All the defined `step`s assume they are being ran as a Valohai "Distributed" task

Your installation also needs to have been configured to place the workers in a subnet with a firewall/security 
group that allows inter-subnet communication.

Don't hesitate to contact your Valohai support with any questions!
