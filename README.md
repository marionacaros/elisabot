# A generative dialogue system for reminiscence therapy

In order to achieve an AI solution to encourage the communication and improve the quality of life of the people affected by cognitive impairment or the onset of Alzheimer's, we introduce Elisabot, an end-to-end system designed to guide older adults through reminiscence sessions with photos from the user.

We implement a conversational agent composed of two deep learning architectures to recognize image and text content. An Encoder-Decoder with Attention is trained to generate questions based on the photos provided by the users. Which is composed of a pretrained Convolution Neural Network to encode the picture, and a Long Short-Term Memory to decode the image features and generate the question. The second architecture is a sequence-to-sequence model that provides feedback to engage the user in the conversation.

Thanks to the experiments, we realise that we obtain the best performance by training it with Persona-Dataset and fine-tune it with Cornell Movie-Dialogues dataset. Finally, we integrate Telegram as the interface for the user to interact with the conversational agent.


## Steps to run the code 

1- Install src/requirements.txt
