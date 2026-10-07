# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **domain** | One review catalog: `food` or `books` | category, dataset (for a domain) |
| **source domain** | The domain that the model reads for a test user | input domain, auxiliary domain |
| **target domain** | The domain that the model recommends from | output domain, destination |
| **rating** | One star value from 1 to 5 that one user gave one item | score (for a star value), review |
| **score** | The number that a model gives an item to rank it | rating (for a model output) |
| **item** | One product or one book, keyed by its ID | product (for a book), title (for an item) |
| **item ID** | `ProductId` for food and `Id` (the ASIN) for books | title, name |
| **title** | The display name of a book. It is never a key | item ID |
| **shared user** | A user with ratings in both domains | common user, overlapping user |
| **overlap** | The set of shared users | intersection |
| **k-core filter** | The filter that keeps users with enough ratings in each domain and items with enough ratings | pruning, cleaning |
| **cold-start user** | A test user whose target ratings are all held out | new user (for a test user) |
| **held-out rating** | A target rating of a test user that no model sees at fit time | test rating, hidden rating |
| **relevant item** | A held-out item with a rating at or above the relevance threshold | positive, hit (for an item) |
| **user vector** | The row of user factors for one user | embedding, latent vector |
| **item vector** | The row of item factors for one item | embedding |
| **bias** | The global mean, user offset or item offset of a domain | intercept, offset (alone) |
| **map** | The learned function from source user vectors to target user vectors in EMCDR | bridge, transfer function |
| **CMF** | Collective matrix factorisation with one user vector for both domains | joint SVD, shared SVD |
| **EMCDR** | Embedding and mapping: two MFs plus a map | mapping model |
| **baseline** | `popularity` or `itembias` | naive model |
| **model folder** | The folder with `model.json` and `arrays.npz` | checkpoint, pickle |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **load** | Read a raw review file into the canonical rating schema |
| **validate** | Drop bad rows and merge duplicate ratings |
| **filter** | Apply the k-core filter |
| **subsample** | Keep a seeded random subset of shared users after the k-core filter |
| **split** | Select the cold-start users and hold out their target ratings |
| **fit** | Learn biases, vectors and the map from the train data |
| **fold in** | Solve a user vector from a rating history with fixed item vectors |
| **score** | Give each target item a number for one user |
| **rank** | Sort the target items by score |
| **recommend** | Return the top-k ranked items without the items that the user rated |
| **evaluate** | Measure recall, NDCG, MAP, hit rate, RMSE and MAE on the held-out ratings |
