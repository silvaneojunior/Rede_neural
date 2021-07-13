float_type='float32'
var_float_type='float32'
underflow_constant_value=10**(-4)
memory_pre_allocation=False

#Esta flag indica se os Tensors criados ao longo de um while_loop devem ser transferidos da GPU para a CPU (ver documentação do tf.while_loop).
#Se o consumo de memória estiver vindo de fora dos parâmetros do modelo (no input, por exemplo), então esta flag não ajuda muito.
swap_memory_flag=True

XLA_opt=False
XLA_func=False

custom_opt=True
layout_optimizer=True
memory_optimizer=True
constant_folding=True
shape_optimization=True
remapping=True
arithmetic_optimization=True
dependency_optimization=True
loop_optimization=True
function_optimization=True
debug_stripper=False
disable_model_pruning=True
scoped_allocator_optimization=True
pin_to_host_optimization=False
implementation_selector=True
auto_mixed_precision=True
disable_meta_optimizer=False
min_graph_nodes=False