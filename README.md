# A generative dialogue system for reminiscence therapy

In order to achieve an AI solution to encourage the communication and improve the quality of life of the people affected by cognitive impairment or the onset of Alzheimer's, we introduce Elisabot, an end-to-end system designed to guide older adults through reminiscence sessions with photos from the user.

We implement a conversational agent composed of two deep learning architectures to recognize image and text content. An Encoder-Decoder with Attention is trained to generate questions based on the photos provided by the users. Which is composed of a pretrained Convolution Neural Network to encode the picture, and a Long Short-Term Memory to decode the image features and generate the question. The second architecture is a sequence-to-sequence model that provides feedback to engage the user in the conversation.

Thanks to the experiments, we realise that we obtain the best performance by training it with Persona-Dataset and fine-tune it with Cornell Movie-Dialogues dataset. Finally, we integrate Telegram as the interface for the user to interact with the conversational agent.


## Steps to run the Authomatic Reminiscence Therapy in Telegram with trained models 

1.  Install required libriaries by using src/requirements.txt  -> 'pip install -r requirements.txt'
2.  Create a new bot, instructions can be found here: https://core.telegram.org/bots you just have to talk with BotFather to get the token which is required to authorize the bot and send requests to the Bot API.  
    The token is a string along the lines of 110201543:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw.
3.  Once the bot is created, place the token in telegram_bot.py  you can pass it by argument.
4.  Download model checkpoints from https://1drv.ms/u/s!Ah93nVed1CWhgRcuqbIFDMKWFDlX?e=KXMZoL and place the folders in /src directory
5.  Run telegram_bot.py -> python telegram_bot.py --token 110201543:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw 
6.  To start the Reminiscence Therapy, look for the bot you created in Telegram and write the command /start
    

