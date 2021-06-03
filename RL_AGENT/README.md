https://www.tecmint.com/linux-cpu-load-stress-test-with-stress-ng-tool/
# DEEP Q - NETWORK
  We use a neural network to approximate the Q function:
  <p align = 'center'>
    <img src = 'https://miro.medium.com/max/296/1*HQZIdIrHlmsXGEd06zqi2A.png'>
  </p>
  
## The loss function is

### Target: 
  <p align = 'center'> q_target the predicted Q value of taking action in a particular state. </p>
  
### Prediction: 
  <p align = 'center'> q_s_a the value you get when actually taking that action, i.e calculating the value on present reward and choosing the next step that maximises the value.
  
<p align = 'center'>
    <img src = 'https://miro.medium.com/max/700/1*BL0PtvUiWsVkUUh-mhAbhg.png'>
</p>

## Parameter updating:

<p align = 'center'>
    <img src = 'https://miro.medium.com/max/700/1*sAtxRESVIxtGfY42gDP49A.png'>
</p>
  
