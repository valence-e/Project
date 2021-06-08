/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
* 
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
* 
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer.
* 
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen	    Email: pchen72 at asu dot edu 
*                    
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cmath>
#include <iostream>
#include <vector>
#include "constant.h"
#include "formula.h"
#include "NewSubArray.h"
#include "Param.h"


using namespace std;

extern Param *param;

NewSubArray::NewSubArray(InputParameter& _inputParameter, Technology& _tech, MemCell& _cell):
						inputParameter(_inputParameter), tech(_tech), cell(_cell){
	initialized = false;
	readDynamicEnergyArray = writeDynamicEnergyArray = 0;
} 

void NewSubArray::Initialize(int _numRow, int _numCol, double _unitWireRes, int _subSubArrayRow, int _subSubArrayCol){  //initialization module
	
	numRow = _numRow;    //import parameters
	numCol = _numCol;
	unitWireRes = _unitWireRes;
    subSubArrayRow = _subSubArrayRow;
    subSubArrayCol = _subSubArrayCol;
    numSubSubArrayRow = numRow/subSubArrayRow;
    numSubSubArrayCol = numCol/subSubArrayCol;

    SubArray *bigArray = new SubArray(inputParameter, tech, cell);
    bigArray->XNORparallelMode = param->XNORparallelMode;               
    bigArray->XNORsequentialMode = param->XNORsequentialMode;             
    bigArray->BNNparallelMode = param->BNNparallelMode;                
    bigArray->BNNsequentialMode = param->BNNsequentialMode;              
    bigArray->conventionalParallel = param->conventionalParallel;                  
    bigArray->conventionalSequential = param->conventionalSequential;                 
    bigArray->numRow = numRow;
    bigArray->numCol = numCol;
    bigArray->levelOutput = param->levelOutput;
    bigArray->numColMuxed = param->numColMuxed;               // How many columns share 1 read circuit (for neuro mode with analog RRAM) or 1 S/A (for memory mode or neuro mode with digital RRAM)
    bigArray->clkFreq = param->clkFreq;                       // Clock frequency
    bigArray->relaxArrayCellHeight = param->relaxArrayCellHeight;
    bigArray->relaxArrayCellWidth = param->relaxArrayCellWidth;
    bigArray->numReadPulse = param->numBitInput;
    bigArray->avgWeightBit = param->cellBit;
    bigArray->numCellPerSynapse = param->numColPerSynapse;
    bigArray->SARADC = param->SARADC;
    bigArray->currentMode = param->currentMode;
    bigArray->validated = param->validated;
    bigArray->spikingMode = NONSPIKING;

    if (bigArray->numColMuxed > numCol) {                      // Set the upperbound of numColMuxed
        bigArray->numColMuxed = numCol;
    }

    bigArray->numReadCellPerOperationFPGA = numCol;	           // Not relevant for IMEC
    bigArray->numWriteCellPerOperationFPGA = numCol;	       // Not relevant for IMEC
    bigArray->numReadCellPerOperationMemory = numCol;          // Define # of SRAM read cells in memory mode because SRAM does not have S/A sharing (not relevant for IMEC)
    bigArray->numWriteCellPerOperationMemory = numCol/8;       // # of write cells per operation in SRAM memory or the memory mode of multifunctional memory (not relevant for IMEC)
    bigArray->numReadCellPerOperationNeuro = numCol;           // # of SRAM read cells in neuromorphic mode
    bigArray->numWriteCellPerOperationNeuro = numCol;	       // For SRAM or analog RRAM in neuro mode
    bigArray->maxNumWritePulse = MAX(cell.maxNumLevelLTP, cell.maxNumLevelLTD);
    bigArray->Initialize(numRow, numCol, unitWireRes, 0, 0);

    resCellAccess = bigArray->resCellAccess;
    capCellAccess = bigArray->capCellAccess;

    subSubArray.reserve(numSubSubArrayRow*numSubSubArrayCol);
    for (int i = 0; i < numSubSubArrayRow; i++) {
        for (int j = 0; j < numSubSubArrayCol; j++) {
            subSubArray.push_back(SubArray(inputParameter, tech, cell));
            subSubArray[i*numSubSubArrayCol+j].XNORparallelMode = param->XNORparallelMode;               
            subSubArray[i*numSubSubArrayCol+j].XNORsequentialMode = param->XNORsequentialMode;             
            subSubArray[i*numSubSubArrayCol+j].BNNparallelMode = param->BNNparallelMode;                
            subSubArray[i*numSubSubArrayCol+j].BNNsequentialMode = param->BNNsequentialMode;              
            subSubArray[i*numSubSubArrayCol+j].conventionalParallel = param->conventionalParallel;                  
            subSubArray[i*numSubSubArrayCol+j].conventionalSequential = param->conventionalSequential;                 
            subSubArray[i*numSubSubArrayCol+j].numRow = numSubSubArrayRow;
            subSubArray[i*numSubSubArrayCol+j].numCol = numSubSubArrayCol;
            subSubArray[i*numSubSubArrayCol+j].levelOutput = param->levelOutput;
            subSubArray[i*numSubSubArrayCol+j].numColMuxed = param->numColMuxed;               // How many columns share 1 read circuit (for neuro mode with analog RRAM) or 1 S/A (for memory mode or neuro mode with digital RRAM)
            subSubArray[i*numSubSubArrayCol+j].clkFreq = param->clkFreq;                       // Clock frequency
            subSubArray[i*numSubSubArrayCol+j].relaxArrayCellHeight = param->relaxArrayCellHeight;
            subSubArray[i*numSubSubArrayCol+j].relaxArrayCellWidth = param->relaxArrayCellWidth;
            subSubArray[i*numSubSubArrayCol+j].numReadPulse = param->numBitInput;
            subSubArray[i*numSubSubArrayCol+j].avgWeightBit = param->cellBit;
            subSubArray[i*numSubSubArrayCol+j].numCellPerSynapse = param->numColPerSynapse;
            subSubArray[i*numSubSubArrayCol+j].SARADC = param->SARADC;
            subSubArray[i*numSubSubArrayCol+j].currentMode = param->currentMode;
            subSubArray[i*numSubSubArrayCol+j].validated = param->validated;
            subSubArray[i*numSubSubArrayCol+j].spikingMode = NONSPIKING;

            if (subSubArray[i*numSubSubArrayCol+j].numColMuxed > numSubSubArrayRow) {                      // Set the upperbound of numColMuxed
                subSubArray[i*numSubSubArrayCol+j].numColMuxed = numSubSubArrayCol;
            }

            subSubArray[i*numSubSubArrayCol+j].numReadCellPerOperationFPGA = numSubSubArrayCol;	           // Not relevant for IMEC
            subSubArray[i*numSubSubArrayCol+j].numWriteCellPerOperationFPGA = numSubSubArrayCol;	       // Not relevant for IMEC
            subSubArray[i*numSubSubArrayCol+j].numReadCellPerOperationMemory = numSubSubArrayCol;          // Define # of SRAM read cells in memory mode because SRAM does not have S/A sharing (not relevant for IMEC)
            subSubArray[i*numSubSubArrayCol+j].numWriteCellPerOperationMemory = numSubSubArrayCol/8;       // # of write cells per operation in SRAM memory or the memory mode of multifunctional memory (not relevant for IMEC)
            subSubArray[i*numSubSubArrayCol+j].numReadCellPerOperationNeuro = numSubSubArrayCol;           // # of SRAM read cells in neuromorphic mode
            subSubArray[i*numSubSubArrayCol+j].numWriteCellPerOperationNeuro = numSubSubArrayCol;	       // For SRAM or analog RRAM in neuro mode
            subSubArray[i*numSubSubArrayCol+j].maxNumWritePulse = MAX(cell.maxNumLevelLTP, cell.maxNumLevelLTD);

        }
    }

    for (int i = 0; i < numSubSubArrayRow; i++) {
        for (int j = 0; j < numSubSubArrayCol; j++) {
            subSubArray[i*numSubSubArrayCol+j].Initialize(subSubArrayRow, subSubArrayCol, unitWireRes, i, j);
        }
    }

    initialized = true;
}

void NewSubArray::PrintId() {
    for (int i = 0; i < numSubSubArrayRow; i++) {
        for (int j = 0; j < numSubSubArrayCol; j++) {
            cout << subSubArray[i*numSubSubArrayCol+j].idX << "," << subSubArray[i*numSubSubArrayCol+j].idY << endl;
        }
    }
}

void NewSubArray::CalculateArea() {  //calculate layout area for total design
    height = 0;
    width = 0;
    usedArea = 0;
	for (int i = 0; i < numSubSubArrayRow; i++) {
        for (int j = 0; j < numSubSubArrayCol; j++) {
            subSubArray[i*numSubSubArrayCol+j].CalculateArea();

            if (i == 0 && j == 0) subSubArray[i*numSubSubArrayCol+j].ValidateArea();

            if (i == 0) { // First Row Includes Top Peripherals
                height += subSubArray[i*numSubSubArrayCol+j].topPeripheralHeight;
                usedArea += subSubArray[i*numSubSubArrayCol+j].topPeripheralArea;
            }
            else if (i == numSubSubArrayRow-1) { // Last Row Includes Bottom Peripherals
                height += subSubArray[i*numSubSubArrayCol+j].botPeripheralHeight;
                usedArea += subSubArray[i*numSubSubArrayCol+j].botPeripheralArea;
            }
            if (j == 0) { // First Column Includes Left Peripherals
                width += subSubArray[i*numSubSubArrayCol+j].leftPeripheralWidth;
                usedArea += subSubArray[i*numSubSubArrayCol+j].leftPeripheralArea;
            }

            height += subSubArray[i*numSubSubArrayCol+j].heightArray;
            width += subSubArray[i*numSubSubArrayCol+j].widthArray;
            heightArray += subSubArray[i*numSubSubArrayCol+j].heightArray;
            widthArray += subSubArray[i*numSubSubArrayCol+j].widthArray;
            usedArea += subSubArray[i*numSubSubArrayCol+j].areaArray;
        }
    }
    areaArray = heightArray * widthArray;
    area = height * width;
    emptyArea = area - usedArea;
}

void NewSubArray::CalculateLatency(double columnRes, const vector<double> &columnResistance, bool CalculateclkFreq) {   //calculate latency for different mode 
    readLatency = 0;
    readLatencyADC = 0;
    readLatencyAccum = 0;
    writeLatency = 0;

    topPeripheralLatency = 0;
    botPeripheralLatency = 0;
    leftPeripheralLatency = 0;
    arrayLatency = 0;

    for (int i = 0; i < numSubSubArrayRow; i++) {
        for (int j = 0; j < numSubSubArrayCol; j++) {
            subSubArray[i*numSubSubArrayCol+j].CalculateLatency(columnRes, columnResistance, CalculateclkFreq);

            if (i == 0 && j == 0) subSubArray[i*numSubSubArrayCol+j].ValidateLatency();

            if (i == 0) { // First Row Includes Top Peripherals
                topPeripheralLatency += subSubArray[i*numSubSubArrayCol+j].topPeripheralLatency;
            }
            else if (i == numSubSubArrayRow-1) { // Last Row Includes Bottom Peripherals
                botPeripheralLatency += subSubArray[i*numSubSubArrayCol+j].botPeripheralLatency;
                readLatencyADC += subSubArray[i*numSubSubArrayCol+j].readLatencyADC;
                readLatencyAccum += subSubArray[i*numSubSubArrayCol+j].readLatencyAccum;
            }
            if (j == 0) { // First Column Includes Left Peripherals
                leftPeripheralLatency += subSubArray[i*numSubSubArrayCol+j].leftPeripheralLatency;
            }

            arrayLatency += subSubArray[i*numSubSubArrayCol+j].arrayLatency;
        }
    }

    readLatency += topPeripheralLatency;
    readLatency += botPeripheralLatency;
    readLatency += leftPeripheralLatency;
    readLatency += arrayLatency;
}

// void NewSubArray::CalculatePower(const vector<double> &columnResistance) {

// }

// void NewSubArray::PrintProperty() {

// }

