import React, { Component } from 'react';
import styled from 'styled-components';
import './App.css';
// import {Keywords} from 'react-marker'
document.body.style.background = "#bfd6f6";

const BTN = styled.button`
  background-color: gray;
  display: inline-block;
  font-size: 1em;
  border: 2px solid whitesmoke;
  border-radius: 3px;
  display: block;
  margin-right: 100px;
  width:100px ;
  height: 40px;
  background-color: black;
  box-shadow: 40px;
  color: whitesmoke;

`;
const Title = styled.h1`
  font-size: 1.5em;
  text-align: left;
  padding-right:3em;
  color: white;
`;
const Wrapper = styled.section`
  background:   black;
  padding: 2em;
  box-shadow: 60px;
  padding-left: 10px;
  padding-right: 10px;
`;

const Input = styled.input`
  padding: 0.5em;
  margin: 0.5em;
  color: black;
  background: white ;
  border: none;
  border-radius: 3px;
  width: 1135px;
  margin-right: 670px;
  height: 100px;
  font-weight  : bold;
`;

const insert = styled.text`
text-align: left;
`;
class App extends Component {
    state = {
        characters: [],
        valueInput: "",
        data: [],
        data2: [],
        colors: [],
        modeldata: [],
        flag: false,
        print: [],
        printScores :[],
        Num_of_entities:[]
    };

    removeCharacter = index => {
        const { characters } = this.state;

        this.setState({
            characters: characters.filter((character, i) => {
                return i !== index;
            })
        });
    }

    handleSubmit = character => {
        this.setState({ characters: [...this.state.characters, character] });
    }




    dorgetter = () => {
        let m = [...this.state.modeldata];
        let data = [...this.state.data];
        var words = []
        var colors2 = []
        var num_entities = 0;
        var sentesce = "";
        var print_scores = []
        for (let i = 0; i < data.length; i++) {
            sentesce += data[i]
        }
        words = sentesce.split(" ");
        for (let i = 0; i < m.length; i++) {
            var Type = m[i].entity
            var color = this.entity2color(Type)
            colors2[i] = color
        }
        this.setState({ colors: [...colors2] })
        this.setState({ data2: [...words] })
        let indents = [];

        print_scores.push(
        <h1>
            Probabilities:

       </h1>);
        for (var i = 0; i < this.state.data2.length; i++) {
            if (m[i] != null &&m[i].entity != 'O') {
                
                console.log(this.state.colors[i]);
                num_entities += 1;               
                print_scores.push(<ul><mark style={{ background: this.state.colors[i], padding: '0.45em 0.6em', margin: '0 0.25em', lineHeight: '1', borderRadius: '0.35em' }}>
                {this.state.data2[i] + "\t" + m[i].score + "\n" }
                <span style={{ lineHeight: '1', borderRadius: '0.35em', verticalAlign: 'middle', marginLeft: '0.5rem', fontSize: '0.8em', fontWeight: 'bold' }} key>{m[i].entity}</span>
            </mark></ul>);

                indents.push(<mark style={{ background: this.state.colors[i], padding: '0.45em 0.6em', margin: '0 0.25em', lineHeight: '1', borderRadius: '0.35em' }}>
                    {this.state.data2[i]}
                    <span style={{ lineHeight: '1', borderRadius: '0.35em', verticalAlign: 'middle', marginLeft: '0.5rem', fontSize: '0.8em', fontWeight: 'bold' }} key>{m[i].entity}</span>
                </mark>);
            }

            else indents.push(this.state.data2[i] + " ");

        }
        this.setState({ print: [...indents] });
        
        this.setState({ printScores: [...print_scores]});
        this.setState({Num_of_entities:num_entities});
        //this.setState({print : [...indents]})
      
    }


    entity2color(val) {
        switch (val) {
            case 'B-geo':
                return "orange";
                break;
            case 'O':
                return "";
                break;
            case 'B-per':
                return "#aa9cfc";
                break;
            case 'I-geo':
                return "yellow";
                break;
            case 'B-gpe':
                return "green"
                break;
            case 'I-org':
                return "red";
                break;
            case 'B-tim':
                return "#00ffff"
                break;
            default:
                return 'white';
        }
    }

    sendHttp = () => {
        const requestOptions = {
            method: 'POST',
            mode: 'no-cors',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: this.state.valueInput })
        };

        fetch('http://127.0.0.1:105/new_code', requestOptions)
            .then(response => response.text())
    }

    getData = () => {
        var text = require('./RawText.json'); //(with path)
        var jsontext = require('./textJSON.json'); //(with path)
        this.setState({ data: text })
        this.setState({ modeldata: jsontext })
        this.dorgetter()

    }

    handleChange = (c) => {
        this.setState({ valueInput: c.target.value })
    }


    render() {

        const { print } = this.state;
        const {printScores} = this.state;
        const {Num_of_entities} = this.state;
        return (
            <div className="App">
                <Wrapper>
                    <Title>
                        Welcome to NER with
                        <mark style={{ background: "#cbdadb", padding: '0.45em 0.6em', margin: '0 0.25em', lineHeight: '1', borderRadius: '0.35em' }}>
                            BERT
                            <span style={{ lineHeight: '1', borderRadius: '0.35em', verticalAlign: 'middle', marginLeft: '0.5rem', fontSize: '0.8em', fontWeight: 'bold' }}>MODEL</span>
                        </mark>
                      
                    </Title>
                </Wrapper>
                 <Input type="text" id="fname" name="fname" value={this.state.valueInput} onChange={(e) => { this.handleChange(e) }} />
                 <BTN onClick={this.sendHttp}>Submit</BTN>
                <BTN onClick={this.getData} type="submit">Result</BTN>
                <div>
                    {print}
                </div>
                <div>
                    <h>
                    Number of Entities = {Num_of_entities}
                    </h>
                </div>

                <div>
                   {printScores}
                </div>
            
            </div>

        );
    
}
}
export default App;