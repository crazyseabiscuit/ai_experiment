# -*- coding: utf-8 -*-
"""ParlAI Tutorial

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bRMvN0lGXaTF5fuTidgvlAl-Lb41F7AD

<img src="https://parl.ai/docs/_static/img/parlai.png" width="700"/>

**Author**: Stephen Roller ([GitHub](https://github.com/stephenroller), [Twitter](https://twitter.com/stephenroller))


# Welcome to the ParlAI interactive tutorial

In this tutorial we will:

- Chat with a neural network model!
- Show how to use common commands in ParlAI, like inspecting data and model outputs.
- See where to find information about many options.
- Show how to fine-tune a pretrained model on a specific task
- Add our own datasets to ParlAI
- And add our own models to ParlAI

We won't be running any examples of using Amazon Mechanical Turk, or connecting to Chat services, but you can check out our [docs](https://parl.ai/docs/) for more information on these areas.

**Note:** *Make sure you're running this session with a GPU attached.*
"""

!nvidia-smi

"""## Installing parlai

We need to install ParlAI. Since we're in Google Colab, we can assume PyTorch and similar dependencies are installed already
"""

!pip install -q parlai
!pip install -q subword_nmt # extra requirement we need for this tutorial

"""# Chatting with a model

Let's start by chatting interactively with a model file from our model zoo! We'll pick our "tutorial transformer generator" model, which is a generative transformer trained on pushshift.io Reddit. You can take a look at the [model zoo](https://parl.ai/docs/zoo.html) for a more complete list.
"""

# Import the Interactive script
from parlai.scripts.interactive import Interactive

# call it with particular args
Interactive.main(
    # the model_file is a filename path pointing to a particular model dump.
    # Model files that begin with "zoo:" are special files distributed by the ParlAI team.
    # They'll be automatically downloaded when you ask to use them.
    model_file='zoo:tutorial_transformer_generator/model'
)

"""The same on the command line:
```bash
python -m parlai.scripts.interactive --model-file zoo:tutorial_transformer_generator/model
```

# Taking a look at some data

We can look at look into a specific dataset. Let's look into the "empathetic dialogues" dataset, which aims to teach models how to respond with text expressing the appropriate emotion. We have over existing 80 datasets in ParlAI. You can take a full look in our [task list](https://parl.ai/docs/tasks.html).
"""

# The display_data script is used to show the contents of a particular task.
# By default, we show the train
from parlai.scripts.display_data import DisplayData
DisplayData.main(task='empathetic_dialogues', num_examples=5)

"""The black, unindented text is the _prompt_, while the blue text is the _label_. That is, the label is what we will be training the model to mimic.

We can also ask to see fewer examples, and get them from the validation set instead.
"""

# we can instead ask to see fewer examples, and get them from the valid set.
DisplayData.main(task='empathetic_dialogues', num_examples=3, datatype='valid')

"""On the command line:
```bash
python -m parlai.scripts.display_data --task empathetic_dialogues
```
or a bit shorter
```
python -m parlai.scripts.display_data -t empathetic_dialogues
```

# Training a model

Well it's one thing looking at data, but what if we want to train our own model (from scratch)? Let's train a very simple seq2seq LSTM with attention, to respond to empathetic dialogues.

To get some extra performance, we'll initialize using GloVe embeddings, but we will cap the training time to 2 minutes for this tutorial. It won't perform very well, but that's okay.
"""

# we'll save it in the "from_scratch_model" directory
!rm -rf from_scratch_model
!mkdir -p from_scratch_model

from parlai.scripts.train_model import TrainModel
TrainModel.main(
    # we MUST provide a filename
    model_file='from_scratch_model/model',
    # train on empathetic dialogues
    task='empathetic_dialogues',
    # limit training time to 2 minutes, and a batchsize of 16
    max_train_time=2 * 60,
    batchsize=16,
    
    # we specify the model type as seq2seq
    model='seq2seq',
    # some hyperparamter choices. We'll use attention. We could use pretrained
    # embeddings too, with embedding_type='fasttext', but they take a long
    # time to download.
    attention='dot',
    # tie the word embeddings of the encoder/decoder/softmax.
    lookuptable='all',
    # truncate text and labels at 64 tokens, for memory and time savings
    truncate=64,
)

"""Our perplexity and F1 (word overlap) scores are pretty bad, and our BLEU-4 score is nearly 0. That's okay, we would normally want to train for well over an hour. Feel free to change the max_train_time above.

## Performance is pretty bad there. Can we improve it?

The easiest way to improve it is to *initialize* using a *pretrained model*, utilizing *transfer learning*. Let's use the one from the interactive session at the beginning of the chat!
"""

!rm -rf from_pretrained
!mkdir -p from_pretrained

TrainModel.main(
    # similar to before
    task='empathetic_dialogues', 
    model='transformer/generator',
    model_file='from_pretrained/model',
    
    # initialize with a pretrained model
    init_model='zoo:tutorial_transformer_generator/model',
    
    # arguments we get from the pretrained model.
    # Unfortunately, these must be looked up separately for each model.
    n_heads=16, n_layers=8, n_positions=512, text_truncate=512,
    label_truncate=128, ffn_size=2048, embedding_size=512,
    activation='gelu', variant='xlm',
    dict_lower=True, dict_tokenizer='bpe',
    dict_file='zoo:tutorial_transformer_generator/model.dict',
    learn_positional_embeddings=True,
    
    # some training arguments, specific to this fine-tuning
    # use a small learning rate with ADAM optimizer
    lr=1e-5, optimizer='adam',
    warmup_updates=100,
    # early stopping on perplexity
    validation_metric='ppl',
    # train at most 10 minutes, and validate every 0.25 epochs
    max_train_time=600, validation_every_n_epochs=0.25,
    
    # depend on your gpu. If you have a V100, this is good
    batchsize=12, fp16=True, fp16_impl='mem_efficient',
    
    # speeds up validation
    skip_generation=True,
    
    # helps us cram more examples into our gpu at a time
    dynamic_batching='full',
)

"""## Wow that's a lot of options? Where do I find more info?

As you might have noticed, there are a LOT of options to ParlAI. You're best reading the [ParlAI docs](https://parl.ai/docs) to find a list of hyperparameters. We provide lists of the command-line args for both models

You can get some guidance in this notebook by using:
"""

# note that if you want to see model-specific arguments, you must specify a model name
print(TrainModel.help(model='seq2seq'))

"""You'll notice the options are give as commandline arguments. We control our options via `argparse`. The option names are relatively predictable: `--init-model` becomes `init_model`; `--num-epochs` becomes `num_epochs` and so on.

# Looking at model predictions

We have shown how we can chat with a model ourselves, interactively. We might want to inspect how the model reacts with a fixed set of inputs. Let's use that model we just trained!
"""

from parlai.scripts.display_model import DisplayModel
DisplayModel.main(
    task='empathetic_dialogues',
    model_file='from_pretrained/model',
    num_examples=2,
)

"""Whoa wait a second! The model isn't giving any responses? That's because we set `--skip-generation true` to speed up training. We need to turn that back off."""

from parlai.scripts.display_model import DisplayModel
DisplayModel.main(
    task='empathetic_dialogues',
    model_file='from_pretrained/model',
    num_examples=2,
    skip_generation=False,
)

"""On the command line:
```bash
python -m parlai.scripts.display_model --task empathetic_dialogues --model-file zoo:tutorial_transformer_generator/model
```

# Bringing your own datasets

What if you want to build your own dataset in ParlAI? Of course you can do that!
"""

from parlai.core.teachers import register_teacher, DialogTeacher

@register_teacher("my_teacher")
class MyTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # opt is the command line arguments.
        
        # What is this shared thing?
        # We make many copies of a teacher, one-per-batchsize. Shared lets us store 
        
        # We just need to set the "datafile".  This is boilerplate, but differs in many teachers.
        # The "datafile" is the filename where we will load the data from. In this case, we'll set it to
        # the fold name (train/valid/test) + ".txt"
        opt['datafile'] = opt['datatype'].split(':')[0] + ".txt"
        super().__init__(opt, shared)
    
    def setup_data(self, datafile):
        # filename tells us where to load from.
        # We'll just use some hardcoded data, but show how you could read the filename here:
        print(f" ~~ Loading from {datafile} ~~ ")
        
        # setup_data should yield tuples of ((text, label), new_episode)
        # That is ((str, str), bool)
        
        # first episode
        # notice how we have call, response, and then True? The True indicates this is a first message
        # in a conversation
        yield ('Hello', 'Hi'), True
        # Next we have the second turn. This time, the last element is False, indicating we're still going
        yield ('How are you', 'I am fine'), False
        yield ("Let's say goodbye", 'Goodbye!'), False
        
        # second episode. We need to have True again!
        yield ("Hey", "hi there"), True
        yield ("Deja vu?", "Deja vu!"), False
        yield ("Last chance", "This is it"), False
        
        
DisplayData.main(task="my_teacher")

"""Notice how the data corresponds to the utterances we provided? In reality, we'd normally want to load up a data file, loop through it, and yield the tuples from processed data. But for this simple example, it works well.

We can now use our teacher in the standard places! Let's see how the model we trained earlier behaves with it:
"""

DisplayModel.main(task='my_teacher', model_file='from_pretrained/model', skip_generation=False)

"""Note that the `register_teacher` decorator makes the commands aware of your teacher. If you leave it off, the commands won't be able to locate it. If you want to use your teacher on the command line, you'll need to put it in a very specific filename: `parlai/agents/my_teacher/agents.py`, and you'll need to name the class `DefaultTeacher` instead of `MyTeacher`.

# Creating your own models

As a start, we'll implement a *very* simple agent. This agent will just sort of respond with "hello X, my name is Y", where X is based on the input
"""

from parlai.core.agents import register_agent, Agent

@register_agent("hello")
class HelloAgent(Agent):
    @classmethod
    def add_cmdline_args(cls, parser, partial_opt):
        parser.add_argument('--name', type=str, default='Alice', help="The agent's name.")
        return parser
        
    def __init__(self, opt, shared=None):
        # similar to the teacher, we have the Opt and the shared memory objects!
        super().__init__(opt, shared)
        self.id = 'HelloAgent'
        self.name = opt['name']
    
    def observe(self, observation):
        # Gather the last word from the other user's input
        words = observation.get('text', '').split()
        if words:
            self.last_word = words[-1]
        else:
            self.last_word = "stranger!"
    
    def act(self):
        # Always return a string like this.
        return {
            'id': self.id,
            'text': f"Hello {self.last_word}, I'm {self.name}",
        }

"""Let's try seeing how this agent behaves:"""

DisplayModel.main(task='my_teacher', model='hello')

"""Notice how it read the words from the user, and provides its name from the command line argument? We can also interact with it easily enough."""

Interactive.main(model='hello', name='Bob')

"""Similar to the teacher, the call to `register_agent` makes it available for use in commands. If you forget the `register_agent` decorator, you won't be able to refer to it. Similarly, if you wanted to use this model from the command line, you would need to save this code to a special folder: `parlai/agents/hello/hello.py`.

## Creating a neural network model

The base Agent class is very simple, but it also provides extremely little functionality. We have created solid abstractions for creating neural-network type models. [`TorchGeneratorAgent`](https://parl.ai/docs/torch_agent.html#module-parlai.core.torch_generator_agent) is one our common abstractions, and it assumes a model which outputs one-word-at-a-time.

The following is from our [ExampleSeq2Seq](https://github.com/facebookresearch/ParlAI/blob/master/parlai/agents/examples/seq2seq.py) agent. It's a simple RNN model, trained like a Machine Translation model. The Model is too complex to go over in this document, but please feel free to [read our TorchGeneratorAgent tutorial](https://parl.ai/docs/tutorial_torch_generator_agent.html).
"""

import torch.nn as nn
import torch.nn.functional as F
import parlai.core.torch_generator_agent as tga


class Encoder(nn.Module):
    """
    Example encoder, consisting of an embedding layer and a 1-layer LSTM with the
    specified hidden size.
    Pay particular attention to the ``forward`` output.
    """

    def __init__(self, embeddings, hidden_size):
        """
        Initialization.
        Arguments here can be used to provide hyperparameters.
        """
        # must call super on all nn.Modules.
        super().__init__()

        self.embeddings = embeddings
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, input_tokens):
        """
        Perform the forward pass for the encoder.
        Input *must* be input_tokens, which are the context tokens given
        as a matrix of lookup IDs.
        :param input_tokens:
            Input tokens as a bsz x seqlen LongTensor.
            Likely will contain padding.
        :return:
            You can return anything you like; it is will be passed verbatim
            into the decoder for conditioning. However, it should be something
            you can easily manipulate in ``reorder_encoder_states``.
            This particular implementation returns the hidden and cell states from the
            LSTM.
        """
        embedded = self.embeddings(input_tokens)
        _output, hidden = self.lstm(embedded)
        return hidden


class Decoder(nn.Module):
    """
    Basic example decoder, consisting of an embedding layer and a 1-layer LSTM with the
    specified hidden size. Decoder allows for incremental decoding by ingesting the
    current incremental state on each forward pass.
    Pay particular note to the ``forward``.
    """

    def __init__(self, embeddings, hidden_size):
        """
        Initialization.
        Arguments here can be used to provide hyperparameters.
        """
        super().__init__()
        self.embeddings = embeddings
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, input, encoder_state, incr_state=None):
        """
        Run forward pass.
        :param input:
            The currently generated tokens from the decoder.
        :param encoder_state:
            The output from the encoder module.
        :parm incr_state:
            The previous hidden state of the decoder.
        """
        embedded = self.embeddings(input)
        if incr_state is None:
            # this is our very first call. We want to seed the LSTM with the
            # hidden state of the decoder
            state = encoder_state
        else:
            # We've generated some tokens already, so we can reuse the existing
            # decoder state
            state = incr_state

        # get the new output and decoder incremental state
        output, incr_state = self.lstm(embedded, state)

        return output, incr_state


class ExampleModel(tga.TorchGeneratorModel):
    """
    ExampleModel implements the abstract methods of TorchGeneratorModel to define how to
    re-order encoder states and decoder incremental states.
    It also instantiates the embedding table, encoder, and decoder, and defines the
    final output layer.
    """

    def __init__(self, dictionary, hidden_size=1024):
        super().__init__(
            padding_idx=dictionary[dictionary.null_token],
            start_idx=dictionary[dictionary.start_token],
            end_idx=dictionary[dictionary.end_token],
            unknown_idx=dictionary[dictionary.unk_token],
        )
        self.embeddings = nn.Embedding(len(dictionary), hidden_size)
        self.encoder = Encoder(self.embeddings, hidden_size)
        self.decoder = Decoder(self.embeddings, hidden_size)

    def output(self, decoder_output):
        """
        Perform the final output -> logits transformation.
        """
        return F.linear(decoder_output, self.embeddings.weight)

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states to select only the given batch indices.
        Since encoder_state can be arbitrary, you must implement this yourself.
        Typically you will just want to index select on the batch dimension.
        """
        h, c = encoder_states
        return h[:, indices, :], c[:, indices, :]

    def reorder_decoder_incremental_state(self, incr_state, indices):
        """
        Reorder the decoder states to select only the given batch indices.
        This method can be a stub which always returns None; this will result in the
        decoder doing a complete forward pass for every single token, making generation
        O(n^2). However, if any state can be cached, then this method should be
        implemented to reduce the generation complexity to O(n).
        """
        h, c = incr_state
        return h[:, indices, :], c[:, indices, :]


@register_agent("my_first_lstm")
class Seq2seqAgent(tga.TorchGeneratorAgent):
    """
    Example agent.
    Implements the interface for TorchGeneratorAgent. The minimum requirement is that it
    implements ``build_model``, but we will want to include additional command line
    parameters.
    """

    @classmethod
    def add_cmdline_args(cls, argparser, partial_opt):
        """
        Add CLI arguments.
        """
        # Make sure to add all of TorchGeneratorAgent's arguments
        super().add_cmdline_args(argparser)

        # Add custom arguments only for this model.
        group = argparser.add_argument_group('Example TGA Agent')
        group.add_argument(
            '-hid', '--hidden-size', type=int, default=1024, help='Hidden size.'
        )

    def build_model(self):
        """
        Construct the model.
        """

        model = ExampleModel(self.dict, self.opt['hidden_size'])
        # Optionally initialize pre-trained embeddings by copying them from another
        # source: GloVe, fastText, etc.
        self._copy_embeddings(model.embeddings.weight, self.opt['embedding_type'])
        return model

"""Of course, now we can train with our new model. Let's train it on our toy task that we created earlier."""

# of course, we can train the model! Let's Train it on our silly toy task from above
!rm -rf my_first_lstm
!mkdir -p my_first_lstm

TrainModel.main(
    model='my_first_lstm',
    model_file='my_first_lstm/model',
    task='my_teacher',
    batchsize=1,
    validation_every_n_secs=10,
    max_train_time=60,
)

"""Let's see how it does. It should reproduce the data perfectly:"""

DisplayModel.main(model_file='my_first_lstm/model', task='my_teacher')

"""Unsurprisingly, we got perfect accuracy. This is because the data set is only a handful of utterances, and we can perfectly memorize it in this LSTM. Nonetheless, a great success!

# What's next!

The sky's the limit! Be sure to check out our [GitHub](https://github.com/facebookresearch/ParlAI) and [Follow ParlAI on Twitter](https://twitter.com/parlai_parley). We're eager to hear what you are using ParlAI for!

Here are some other great resources:
- [Our research page](https://parl.ai/projects/)
- [ParlAI Documentations](https://parl.ai/docs/index.html)
- [Tutorial: Writing a Ranker model](https://parl.ai/docs/tutorial_torch_ranker_agent.html)
- [Tutorial: Using Mechanical Turk](https://parl.ai/docs/tutorial_mturk.html)
- [Tutorial: Connecting to chat services](https://parl.ai/docs/tutorial_chat_service.html)
"""